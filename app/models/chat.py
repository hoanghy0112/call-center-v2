from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Model for a chat message."""

    role: str  # 'user' or 'assistant'
    content: Any  # Can be a string or structured content


class ChatResponse(BaseModel):
    """Model for a chat response."""

    session_id: str
    text: str
    status: str
    metadata: Optional[Dict[str, Any]] = None
