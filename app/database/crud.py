from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Customer, TeamMember


async def get_customer_by_email(db: AsyncSession, email: str) -> Customer | None:
    result = await db.execute(select(Customer).where(Customer.email == email))
    return result.scalar_one_or_none()


async def get_customer_by_id(db: AsyncSession, customer_id: int) -> Customer | None:
    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    return result.scalar_one_or_none()


async def get_all_customers(db: AsyncSession) -> list[Customer]:
    result = await db.execute(select(Customer))
    return result.scalars().all()


async def create_customer(db: AsyncSession, **kwargs) -> Customer:
    customer = Customer(**kwargs)
    db.add(customer)
    await db.commit()
    await db.refresh(customer)
    return customer


async def update_customer(db: AsyncSession, customer_id: int, **kwargs) -> Customer | None:
    customer = await get_customer_by_id(db, customer_id)
    if not customer:
        return None
    for field, value in kwargs.items():
        setattr(customer, field, value)
    await db.commit()
    await db.refresh(customer)
    return customer


async def delete_customer(db: AsyncSession, customer_id: int) -> bool:
    customer = await get_customer_by_id(db, customer_id)
    if not customer:
        return False
    await db.delete(customer)
    await db.commit()
    return True


# --- TeamMember CRUD ---

async def get_team_member_by_email(db: AsyncSession, email: str) -> TeamMember | None:
    result = await db.execute(select(TeamMember).where(TeamMember.email == email))
    return result.scalar_one_or_none()


async def get_team_member_by_id(db: AsyncSession, member_id: int) -> TeamMember | None:
    result = await db.execute(select(TeamMember).where(TeamMember.id == member_id))
    return result.scalar_one_or_none()


async def get_all_team_members(db: AsyncSession) -> list[TeamMember]:
    result = await db.execute(select(TeamMember))
    return result.scalars().all()


async def create_team_member(db: AsyncSession, **kwargs) -> TeamMember:
    member = TeamMember(**kwargs)
    db.add(member)
    await db.commit()
    await db.refresh(member)
    return member


async def update_team_member(db: AsyncSession, member_id: int, **kwargs) -> TeamMember | None:
    member = await get_team_member_by_id(db, member_id)
    if not member:
        return None
    for field, value in kwargs.items():
        setattr(member, field, value)
    await db.commit()
    await db.refresh(member)
    return member


async def delete_team_member(db: AsyncSession, member_id: int) -> bool:
    member = await get_team_member_by_id(db, member_id)
    if not member:
        return False
    await db.delete(member)
    await db.commit()
    return True
