import uuid
import os
from typing import Any, List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import numpy as np
import json
import time

from app.core.chat_service import ChatService
from app.constants import WAV_DIR
from app.core.db import SessionDep
from app.models.chat import ChatMessage, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/complement")
async def chat_complement(
    audio_file: UploadFile = File(...),
    session_id: str = Form(None),
) -> Any:
    """
    Process an audio file for chat complement.

    Args:
        audio_file: The uploaded audio file
        session_id: Optional session ID to continue a conversation

    Returns:
        Chat response with text and any additional metadata
    """
    try:
        # Generate a unique session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Read the file content
        file_content = await audio_file.read()

        # Save the audio file using the service
        file_path = ChatService.save_audio_file(file_content, session_id)

        # Create a user message for this audio
        user_message = {"type": "audio", "audio": file_path}

        # Save the user message to the conversation history
        ChatService.save_conversation_message(session_id, "user", [user_message])

        # Get the full conversation history
        conversation = ChatService.get_previous_conversation(session_id)

        # Process the conversation with the model
        response_generator = ChatService.process_conversation(conversation)
        response = next(response_generator)

        # Extract the text response
        text_response = response.get("generated_text", "")

        # Save the assistant's response to the conversation history
        ChatService.save_conversation_message(session_id, "assistant", text_response)

        # Create and return the response
        return ChatResponse(session_id=session_id, text=text_response, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.get("/{session_id}/history")
async def get_chat_history(session_id: str) -> Any:
    """
    Get the history of a chat session.

    Args:
        session_id: The session ID

    Returns:
        List of chat messages in the session
    """
    try:
        # Get the conversation for this session
        conversation = ChatService.get_previous_conversation(session_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"session_id": session_id, "messages": conversation, "status": "success"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chat history: {str(e)}"
        )
