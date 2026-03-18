from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database import crud

router = APIRouter(prefix="/teams", tags=["teams"])


class TeamMemberCreate(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str | None = None


class TeamMemberUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone: str | None = None


class TeamMemberResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    phone: str | None

    class Config:
        from_attributes = True


@router.get("/", response_model=list[TeamMemberResponse])
async def list_team_members(db: AsyncSession = Depends(get_db)):
    return await crud.get_all_team_members(db)


@router.get("/{member_id}", response_model=TeamMemberResponse)
async def get_team_member(member_id: int, db: AsyncSession = Depends(get_db)):
    member = await crud.get_team_member_by_id(db, member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Membre introuvable")
    return member


@router.post("/", response_model=TeamMemberResponse, status_code=201)
async def create_team_member(body: TeamMemberCreate, db: AsyncSession = Depends(get_db)):
    existing = await crud.get_team_member_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    return await crud.create_team_member(db, **body.model_dump())


@router.patch("/{member_id}", response_model=TeamMemberResponse)
async def update_team_member(member_id: int, body: TeamMemberUpdate, db: AsyncSession = Depends(get_db)):
    updated = await crud.update_team_member(db, member_id, **body.model_dump(exclude_none=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Membre introuvable")
    return updated


@router.delete("/{member_id}", status_code=204)
async def delete_team_member(member_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_team_member(db, member_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Membre introuvable")
