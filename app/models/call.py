from sqlmodel import SQLModel, Field
from uuid import UUID
from datetime import datetime
from sqlalchemy import func


class Call(SQLModel, table=True):
    id: UUID = Field(default_factory=func.uuid_generate_v4, primary_key=True)
    voice: str = Field(
        default="en_US-bryce-medium", sa_column_kwargs={"default": "en_US-bryce-medium"}
    )
    temperature: float = Field(default=1.0, sa_column_kwargs={"default": 1})
    system_prompt: str = Field(default="", sa_column_kwargs={"default": ""})
    language: str = Field(default="en", sa_column_kwargs={"default": "en"})
    use_lipsync: bool = Field(default=True, sa_column_kwargs={"default": True})
    created_at: datetime = Field(
        default_factory=datetime.utcnow, sa_column_kwargs={"default": func.now()}
    )
