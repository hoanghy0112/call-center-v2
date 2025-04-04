import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, HTTPException
from sqlmodel import func, select

from app.core.db import SessionDep
from app.models.call import Call

router = APIRouter(prefix="/calls", tags=["calls"])


@router.post("/")
def create_call(session: SessionDep, item: Call) -> Any:

    session.add(item)
    session.commit()
    session.refresh(item)

    return {
        "call": item,
        "join_url": f"wss://localhost:8000/calls/{item.id}/web_socket",
    }
