import uuid
from typing import Optional

from app.config import get_settings
from app.data.questions import QUESTIONS_EN, QUESTIONS_FR
from app.models.session import SessionRecord
from app.services.livekit_service import create_token
from app.services import redis_service

_sessions: dict[str, SessionRecord] = {}


def create_session(language: str, voice_gender: str, question_count: int) -> dict:
    settings = get_settings()

    session_id = str(uuid.uuid4())
    room_name = f"room_{session_id}"
    user_identity = f"user_{session_id}"
    agent_identity = f"agent_{session_id}"

    voice_id = settings.voice_mapping[language][voice_gender]
    questions_pool = QUESTIONS_FR if language == "fr" else QUESTIONS_EN
    questions = questions_pool[:question_count]

    user_token = create_token(user_identity, room_name)
    agent_token = create_token(agent_identity, room_name)

    redis_service.save_session_meta(session_id, language, voice_gender)

    _sessions[session_id] = SessionRecord(
        session_id=session_id,
        room_name=room_name,
        language=language,
        voice_gender=voice_gender,
        voice_id=voice_id,
        questions=questions,
        current_index=0,
        answers={},
        agent_token=agent_token,
    )

    return {
        "session_id": session_id,
        "room_name": room_name,
        "token": user_token,
        "livekit_url": settings.livekit_url,
        "identity": user_identity,
    }


def get_session(session_id: str) -> Optional[SessionRecord]:
    return _sessions.get(session_id)


def list_session_ids() -> list[str]:
    return list(_sessions.keys())


def submit_answer(session_id: str, text: str) -> Optional[dict]:
    session = _sessions.get(session_id)
    if session is None:
        return None

    idx = session["current_index"]
    question = session["questions"][idx]

    if text not in question["choices"]:
        msg = (
            f"Invalid answer. Possible choices: {', '.join(question['choices'])}"
            if session["language"] != "fr"
            else f"Réponse invalide. Choix possibles: {', '.join(question['choices'])}"
        )
        return {"action": "RETRY", "text": msg}

    session["answers"][question["id"]] = text
    session["current_index"] += 1

    if session["current_index"] >= len(session["questions"]):
        msg = "Thank you, questionnaire completed." if session["language"] != "fr" else "Merci, questionnaire terminé."
        return {"action": "END", "text": msg}

    next_question = session["questions"][session["current_index"]]
    return {
        "action": "ASK",
        "text": next_question["question"],
        "choices": next_question["choices"],
    }
