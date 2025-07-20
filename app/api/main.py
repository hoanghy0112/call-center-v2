from fastapi import APIRouter

from app.api.routes import calls, chat_complement

api_router = APIRouter()

api_router.include_router(calls.router)
api_router.include_router(chat_complement.router)
