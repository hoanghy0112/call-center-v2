from fastapi import APIRouter

from app.api.routes import calls

api_router = APIRouter()

api_router.include_router(calls.router)
