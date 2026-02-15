import json
from datetime import datetime, timezone

import redis

from app.config import get_settings


_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def save_session_meta(
    session_id: str,
    language: str,
    voice_gender: str,
    voice_id: str,
    room_name: str,
    questions: list,
    agent_token: str,
) -> None:
    r = _get_client()
    r.hset(f"session:{session_id}:meta", mapping={
        "language": language,
        "voice_gender": voice_gender,
        "voice_id": voice_id,
        "room_name": room_name,
        "questions": json.dumps(questions),
        "agent_token": agent_token,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    r.sadd("sessions:index", session_id)


def get_session_meta(session_id: str) -> dict | None:
    r = _get_client()
    meta = r.hgetall(f"session:{session_id}:meta")
    if not meta:
        return None
    if "questions" in meta:
        meta["questions"] = json.loads(meta["questions"])
    return meta


def list_session_ids() -> list[str]:
    r = _get_client()
    return list(r.smembers("sessions:index"))


def save_answer(
    session_id: str,
    question_id: int,
    question_text: str,
    top_2: list[str],
    bottom_2: list[str],
) -> None:
    r = _get_client()
    r.hset(f"session:{session_id}:answers", str(question_id), json.dumps({
        "question": question_text,
        "top_2": top_2,
        "bottom_2": bottom_2,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }))


def get_session_answers(session_id: str) -> dict | None:
    r = _get_client()

    meta = r.hgetall(f"session:{session_id}:meta")
    if not meta:
        return None

    raw_answers = r.hgetall(f"session:{session_id}:answers")
    answers = {
        qid: json.loads(data) for qid, data in raw_answers.items()
    }

    return {
        "session_id": session_id,
        **meta,
        "answers": answers,
    }


def save_user_profile(session_id: str, field: str, value: str) -> None:
    r = _get_client()
    r.hset(f"session:{session_id}:profile", field, value)


def get_user_profile(session_id: str) -> dict | None:
    r = _get_client()
    profile = r.hgetall(f"session:{session_id}:profile")
    if not profile:
        return None
    return profile


REQUIRED_PROFILE_FIELDS = {"first_name", "gender", "age", "has_allergies"}


def is_profile_complete(session_id: str) -> bool:
    r = _get_client()
    profile = r.hgetall(f"session:{session_id}:profile")
    return REQUIRED_PROFILE_FIELDS.issubset(profile.keys())


def get_missing_profile_fields(session_id: str) -> list[str]:
    r = _get_client()
    profile = r.hgetall(f"session:{session_id}:profile")
    return list(REQUIRED_PROFILE_FIELDS - profile.keys())


def get_session_state(session_id: str) -> str:
    if is_profile_complete(session_id):
        return "questionnaire"
    return "collecting_profile"


def get_all_sessions() -> list[dict]:
    r = _get_client()
    session_ids = r.smembers("sessions:index")

    results = []
    for sid in session_ids:
        data = get_session_answers(sid)
        if data:
            results.append(data)

    return results
