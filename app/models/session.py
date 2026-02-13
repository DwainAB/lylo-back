from typing import TypedDict


class Question(TypedDict):
    id: int
    question: str
    choices: list[str]


class SessionRecord(TypedDict):
    session_id: str
    room_name: str
    language: str
    voice_gender: str
    voice_id: str
    questions: list[Question]
    current_index: int
    answers: dict[int, str]
    agent_token: str
