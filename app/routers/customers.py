from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database import crud

router = APIRouter(prefix="/customers", tags=["customers"])


class CustomerCreate(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str | None = None
    days_available: int = 0
    sessions_available: int = 0


class CustomerUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone: str | None = None
    days_available: int | None = None
    sessions_available: int | None = None


class CustomerResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    phone: str | None
    days_available: int
    sessions_available: int

    class Config:
        from_attributes = True


@router.get("/", response_model=list[CustomerResponse])
async def list_customers(db: AsyncSession = Depends(get_db)):
    return await crud.get_all_customers(db)


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: int, db: AsyncSession = Depends(get_db)):
    customer = await crud.get_customer_by_id(db, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Client introuvable")
    return customer


@router.post("/", response_model=CustomerResponse, status_code=201)
async def create_customer(body: CustomerCreate, db: AsyncSession = Depends(get_db)):
    existing = await crud.get_customer_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    return await crud.create_customer(db, **body.model_dump())


@router.patch("/{customer_id}", response_model=CustomerResponse)
async def update_customer(customer_id: int, body: CustomerUpdate, db: AsyncSession = Depends(get_db)):
    updated = await crud.update_customer(db, customer_id, **body.model_dump(exclude_none=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Client introuvable")
    return updated


@router.delete("/{customer_id}", status_code=204)
async def delete_customer(customer_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_customer(db, customer_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Client introuvable")
