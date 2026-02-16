from typing import Literal

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    language: Literal["fr", "en"] = "fr"
    voice_gender: Literal["female", "male"] = "female"
    question_count: int = Field(default=1, ge=1, le=12)


class StartSessionResponse(BaseModel):
    session_id: str
    room_name: str
    token: str
    livekit_url: str
    identity: str


class SaveAnswerRequest(BaseModel):
    question_id: int
    question_text: str
    top_2: list[str]
    bottom_2: list[str]


class SaveProfileRequest(BaseModel):
    field: Literal["first_name", "gender", "age", "has_allergies", "allergies"]
    value: str


class SelectFormulaRequest(BaseModel):
    formula_index: int  # 0 ou 1


class ReplaceNoteRequest(BaseModel):
    note_type: Literal["top", "heart", "base"]
    old_note: str
    new_note: str
