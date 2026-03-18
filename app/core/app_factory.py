from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import mail, sessions, customers, teams, lookup

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Lilo Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sessions.router)
    app.include_router(mail.router)
    app.include_router(customers.router)
    app.include_router(teams.router)
    app.include_router(lookup.router)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    return app
