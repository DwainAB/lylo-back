import unicodedata
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database import crud
from app.models.schemas import (
    ChangeFormulaTypeRequest,
    GenerateFormulasRequest,
    ReplaceNoteRequest,
    SaveAnswerRequest,
    SaveProfileRequest,
    SelectFormulaRequest,
    StartSessionRequest,
    StartSessionResponse,
)
from app.config import get_settings
from app.services import formula_service, livekit_service, mail_service, session_store, session_service

router = APIRouter(prefix="/api", tags=["sessions"])


@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(body: StartSessionRequest, db: AsyncSession = Depends(get_db)):
    if body.email:
        customer = await crud.get_customer_by_email(db, body.email)
        if customer:
            if int(customer.sessions_available) <= 0:
                raise HTTPException(status_code=403, detail="Aucune session disponible")
            max_date = customer.max_date.date() if hasattr(customer.max_date, 'date') else customer.max_date
            if max_date and date.today() > max_date:
                raise HTTPException(status_code=403, detail="Date d'accès expirée")
            await crud.update_customer(db, customer.id, sessions_available=int(customer.sessions_available) - 1)
        else:
            member = await crud.get_team_member_by_email(db, body.email)
            if not member:
                raise HTTPException(status_code=404, detail="Email introuvable")

    try:
        result = await session_service.create_session(
            language=body.language,
            voice_gender=body.voice_gender,
            question_count=body.question_count,
            mode=body.mode,
            input_mode=body.input_mode,
            customer_email=body.email,
            avatar=body.avatar,
        )
    except Exception as e:
        if body.email and customer:
            await crud.update_customer(db, customer.id, sessions_available=int(customer.sessions_available) + 1)
        raise HTTPException(status_code=500, detail=f"Erreur création session: {e}")
    return result


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    meta = session_store.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    room_name = meta.get("room_name", f"room_{session_id}")
    session_store.delete_session(session_id)
    await livekit_service.delete_room(room_name)
    return {"status": "ok", "session_id": session_id}


@router.get("/session_list")
async def session_list():
    return session_service.list_session_ids()


def _normalize(s: str) -> str:
    """Lowercase + supprime les accents pour comparaison souple."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def _canonical_choice(submitted: str, valid_labels: list[str]) -> str:
    """Retourne le label canonique le plus proche parmi valid_labels.

    Ordre de priorité :
    1. Correspondance exacte (insensible à la casse/accents)
    2. Correspondance du préfixe avant " - " (insensible à la casse/accents)
    3. Le label canonique dont le préfixe normalisé contient le texte soumis
    4. Valeur soumise inchangée si aucun match
    """
    norm_submitted = _normalize(submitted)
    # Préfixe soumis (avant " - ")
    submitted_prefix = norm_submitted.split(" - ")[0].strip()

    for label in valid_labels:
        if _normalize(label) == norm_submitted:
            return label

    for label in valid_labels:
        label_prefix = _normalize(label).split(" - ")[0].strip()
        if label_prefix == submitted_prefix:
            return label

    for label in valid_labels:
        label_prefix = _normalize(label).split(" - ")[0].strip()
        if submitted_prefix in label_prefix or label_prefix in submitted_prefix:
            return label

    return submitted


def _normalize_choices(choices: list[str], valid_labels: list[str]) -> list[str]:
    return [_canonical_choice(c, valid_labels) for c in choices]


@router.post("/session/{session_id}/save-answer")
async def save_answer(session_id: str, body: SaveAnswerRequest):
    if not session_store.is_profile_complete(session_id):
        raise HTTPException(
            status_code=400,
            detail="Profile incomplete, cannot save answers yet",
        )

    # Récupérer les choix valides pour cette question depuis la session
    meta = session_store.get_session_meta(session_id)
    top_2 = body.top_2
    bottom_2 = body.bottom_2
    if meta:
        questions = meta.get("questions", [])
        question = next((q for q in questions if q["id"] == body.question_id), None)
        if question:
            valid_labels = [
                c["label"] if isinstance(c, dict) else c
                for c in question.get("choices", [])
            ]
            top_2 = _normalize_choices(top_2, valid_labels)
            bottom_2 = _normalize_choices(bottom_2, valid_labels)

    session_store.save_answer(
        session_id=session_id,
        question_id=body.question_id,
        question_text=body.question_text,
        top_2=top_2,
        bottom_2=bottom_2,
    )
    return {"status": "ok"}


@router.get("/session/{session_id}/answers")
async def get_answers(session_id: str):
    data = session_store.get_session_answers(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found in Redis")
    return data


@router.post("/session/{session_id}/save-profile")
async def save_profile(session_id: str, body: SaveProfileRequest):
    session_store.save_user_profile(session_id, body.field, body.value)
    complete = session_store.is_profile_complete(session_id)
    missing = session_store.get_missing_profile_fields(session_id)
    state = "questionnaire" if complete else "collecting_profile"
    return {
        "status": "ok",
        "state": state,
        "profile_complete": complete,
        "missing_fields": missing,
    }


@router.get("/session/{session_id}/state")
async def get_state(session_id: str):
    state = session_store.get_session_state(session_id)
    complete = session_store.is_profile_complete(session_id)
    missing = session_store.get_missing_profile_fields(session_id)
    mail_available = session_store.get_selected_formula(session_id) is not None
    return {
        "state": state,
        "profile_complete": complete,
        "missing_fields": missing,
        "mail_available": mail_available,
    }


@router.get("/session/{session_id}/profile")
async def get_profile(session_id: str):
    profile = session_store.get_user_profile(session_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("/session/{session_id}/generate-formulas")
async def generate_formulas(session_id: str, body: GenerateFormulasRequest = GenerateFormulasRequest()):
    if not session_store.is_profile_complete(session_id):
        raise HTTPException(
            status_code=400,
            detail="Profile incomplete, cannot generate formulas",
        )
    result = formula_service.generate_formulas(session_id, force_type=body.formula_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


def _send_formula_mail_bg(session_id: str, formula: dict) -> None:
    internal_email = get_settings().internal_email
    if not internal_email:
        return
    recipients = [e.strip() for e in internal_email.split(",") if e.strip()]
    for email in recipients:
        try:
            mail_service.send_mail(email, session_id, formula)
        except Exception as e:
            print(f"[mail] Erreur envoi mail formule à {email} : {e}")
        try:
            mail_service.send_internal_formula_mail(email, session_id, formula)
        except Exception as e:
            print(f"[mail] Erreur envoi mail fiche complète à {email} : {e}")


@router.post("/session/{session_id}/select-formula")
async def select_formula(
    session_id: str, body: SelectFormulaRequest, background_tasks: BackgroundTasks
):
    result = formula_service.select_formula(session_id, body.formula_index)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    background_tasks.add_task(_send_formula_mail_bg, session_id, result["formula"])
    return result


@router.post("/session/{session_id}/change-formula-type")
async def change_formula_type(session_id: str, body: ChangeFormulaTypeRequest):
    result = formula_service.change_selected_formula_type(session_id, body.formula_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/session/{session_id}/available-ingredients/{note_type}")
async def available_ingredients(session_id: str, note_type: str):
    result = formula_service.get_available_ingredients(session_id, note_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/session/{session_id}/replace-note")
async def replace_note(session_id: str, body: ReplaceNoteRequest, background_tasks: BackgroundTasks):
    result = formula_service.replace_note(
        session_id, body.note_type, body.old_note, body.new_note
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    formula = session_store.get_selected_formula(session_id)
    if formula:
        background_tasks.add_task(_send_formula_mail_bg, session_id, formula)
    return result


@router.get("/sessions/all-answers")
async def get_all_answers():
    return session_store.get_all_sessions()
