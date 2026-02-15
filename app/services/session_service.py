import uuid

from app.config import get_settings
from app.data.questions import QUESTIONS_EN, QUESTIONS_FR, _enrich_questions
from app.services.livekit_service import create_token
from app.services import redis_service


def create_session(language: str, voice_gender: str, question_count: int) -> dict:
    settings = get_settings()

    session_id = str(uuid.uuid4())
    room_name = f"room_{session_id}"
    user_identity = f"user_{session_id}"
    agent_identity = f"agent_{session_id}"

    voice_id = settings.voice_mapping[language][voice_gender]
    questions_pool = QUESTIONS_FR if language == "fr" else QUESTIONS_EN
    questions = _enrich_questions(questions_pool[:question_count])

    user_token = create_token(user_identity, room_name)
    agent_token = create_token(agent_identity, room_name)

    redis_service.save_session_meta(
        session_id=session_id,
        language=language,
        voice_gender=voice_gender,
        voice_id=voice_id,
        room_name=room_name,
        questions=questions,
        agent_token=agent_token,
    )

    return {
        "session_id": session_id,
        "room_name": room_name,
        "token": user_token,
        "livekit_url": settings.livekit_url,
        "identity": user_identity,
    }


def get_session(session_id: str) -> dict | None:
    return redis_service.get_session_meta(session_id)


def list_session_ids() -> list[str]:
    return redis_service.list_session_ids()
