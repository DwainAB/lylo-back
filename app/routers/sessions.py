from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ReplaceNoteRequest,
    SaveAnswerRequest,
    SaveProfileRequest,
    SelectFormulaRequest,
    StartSessionRequest,
    StartSessionResponse,
)
from app.services import formula_service, livekit_service, redis_service, session_service

router = APIRouter(prefix="/api", tags=["sessions"])


@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(body: StartSessionRequest):
    result = session_service.create_session(
        language=body.language,
        voice_gender=body.voice_gender,
        question_count=body.question_count,
    )
    return result


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    meta = redis_service.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    room_name = meta.get("room_name", f"room_{session_id}")
    await livekit_service.delete_room(room_name)
    redis_service.delete_session(session_id)
    return {"status": "ok", "session_id": session_id}


@router.get("/session_list")
async def session_list():
    return session_service.list_session_ids()


@router.post("/session/{session_id}/save-answer")
async def save_answer(session_id: str, body: SaveAnswerRequest):
    if not redis_service.is_profile_complete(session_id):
        raise HTTPException(
            status_code=400,
            detail="Profile incomplete, cannot save answers yet",
        )
    redis_service.save_answer(
        session_id=session_id,
        question_id=body.question_id,
        question_text=body.question_text,
        top_2=body.top_2,
        bottom_2=body.bottom_2,
    )
    return {"status": "ok"}


@router.get("/session/{session_id}/answers")
async def get_answers(session_id: str):
    data = redis_service.get_session_answers(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found in Redis")
    return data


@router.post("/session/{session_id}/save-profile")
async def save_profile(session_id: str, body: SaveProfileRequest):
    redis_service.save_user_profile(session_id, body.field, body.value)
    complete = redis_service.is_profile_complete(session_id)
    missing = redis_service.get_missing_profile_fields(session_id)
    state = "questionnaire" if complete else "collecting_profile"
    return {
        "status": "ok",
        "state": state,
        "profile_complete": complete,
        "missing_fields": missing,
    }


@router.get("/session/{session_id}/state")
async def get_state(session_id: str):
    state = redis_service.get_session_state(session_id)
    complete = redis_service.is_profile_complete(session_id)
    missing = redis_service.get_missing_profile_fields(session_id)
    return {
        "state": state,
        "profile_complete": complete,
        "missing_fields": missing,
    }


@router.get("/session/{session_id}/profile")
async def get_profile(session_id: str):
    profile = redis_service.get_user_profile(session_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("/session/{session_id}/generate-formulas")
async def generate_formulas(session_id: str):
    if not redis_service.is_profile_complete(session_id):
        raise HTTPException(
            status_code=400,
            detail="Profile incomplete, cannot generate formulas",
        )
    result = formula_service.generate_formulas(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/session/{session_id}/select-formula")
async def select_formula(session_id: str, body: SelectFormulaRequest):
    result = formula_service.select_formula(session_id, body.formula_index)
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
async def replace_note(session_id: str, body: ReplaceNoteRequest):
    result = formula_service.replace_note(
        session_id, body.note_type, body.old_note, body.new_note
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/sessions/all-answers")
async def get_all_answers():
    return redis_service.get_all_sessions()
