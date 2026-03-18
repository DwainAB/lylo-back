from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database import crud

router = APIRouter(prefix="/lookup", tags=["lookup"])


@router.get("/")
async def lookup_by_email(email: str, db: AsyncSession = Depends(get_db)):
    customer = await crud.get_customer_by_email(db, email)
    if customer:
        return {
            "found": True,
            "table": "customers",
            "id": customer.id,
            "first_name": customer.first_name,
            "last_name": customer.last_name,
            "email": customer.email,
            "phone": customer.phone,
            "days_available": customer.days_available,
            "sessions_available": customer.sessions_available,
        }

    member = await crud.get_team_member_by_email(db, email)
    if member:
        return {
            "found": True,
            "table": "teams",
            "id": member.id,
            "first_name": member.first_name,
            "last_name": member.last_name,
            "email": member.email,
            "phone": member.phone,
        }

    raise HTTPException(status_code=404, detail="Email introuvable dans customers et teams")
