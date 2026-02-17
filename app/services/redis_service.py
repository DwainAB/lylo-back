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


SESSION_TTL_SECONDS = 3600  # 1 hour


def _set_session_ttl(r: redis.Redis, session_id: str) -> None:
    """Apply TTL to all keys belonging to a session."""
    for suffix in ("meta", "answers", "profile", "generated_formulas", "selected_formula"):
        key = f"session:{session_id}:{suffix}"
        if r.exists(key):
            r.expire(key, SESSION_TTL_SECONDS)


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
    r.expire(f"session:{session_id}:meta", SESSION_TTL_SECONDS)


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
    r.expire(f"session:{session_id}:answers", SESSION_TTL_SECONDS)


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
    r.expire(f"session:{session_id}:profile", SESSION_TTL_SECONDS)


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
    if not REQUIRED_PROFILE_FIELDS.issubset(profile.keys()):
        return False
    # If user declared allergies, the actual allergies must also be provided
    if profile.get("has_allergies", "").lower() in ("oui", "yes") and "allergies" not in profile:
        return False
    return True


def get_missing_profile_fields(session_id: str) -> list[str]:
    r = _get_client()
    profile = r.hgetall(f"session:{session_id}:profile")
    missing = list(REQUIRED_PROFILE_FIELDS - profile.keys())
    # If user declared allergies but hasn't specified them yet
    if profile.get("has_allergies", "").lower() in ("oui", "yes") and "allergies" not in profile:
        missing.append("allergies")
    return missing


def get_session_state(session_id: str) -> str:
    if is_profile_complete(session_id):
        return "questionnaire"
    return "collecting_profile"


def save_selected_formula(session_id: str, formula_data: dict) -> None:
    r = _get_client()
    r.set(f"session:{session_id}:selected_formula", json.dumps(formula_data))
    r.expire(f"session:{session_id}:selected_formula", SESSION_TTL_SECONDS)


def get_selected_formula(session_id: str) -> dict | None:
    r = _get_client()
    raw = r.get(f"session:{session_id}:selected_formula")
    if not raw:
        return None
    return json.loads(raw)


def save_generated_formulas(session_id: str, formulas: list[dict]) -> None:
    r = _get_client()
    r.set(f"session:{session_id}:generated_formulas", json.dumps(formulas))
    r.expire(f"session:{session_id}:generated_formulas", SESSION_TTL_SECONDS)


def get_generated_formulas(session_id: str) -> list[dict] | None:
    r = _get_client()
    raw = r.get(f"session:{session_id}:generated_formulas")
    if not raw:
        return None
    return json.loads(raw)


def get_all_sessions() -> list[dict]:
    r = _get_client()
    session_ids = r.smembers("sessions:index")

    results = []
    for sid in session_ids:
        data = get_session_answers(sid)
        if data:
            results.append(data)

    return results


def delete_session(session_id: str) -> bool:
    """Delete all Redis keys for a session. Returns True if the session existed."""
    r = _get_client()
    keys = [
        f"session:{session_id}:meta",
        f"session:{session_id}:answers",
        f"session:{session_id}:profile",
        f"session:{session_id}:generated_formulas",
        f"session:{session_id}:selected_formula",
    ]
    deleted = r.delete(*keys)
    r.srem("sessions:index", session_id)
    return deleted > 0
