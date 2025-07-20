import os
import time
import uuid
from typing import List, Dict, Any, Optional, Generator
import glob

from app.audio_to_text import inference
from app.constants import WAV_DIR
from app.core.conversation_history import ConversationHistoryManager


class ChatService:
    """Service for handling chat operations."""

    @staticmethod
    def save_audio_file(file_content: bytes, session_id: Optional[str] = None) -> str:
        """
        Save an audio file to disk.

        Args:
            file_content: The binary content of the audio file
            session_id: Optional session ID to use in the filename

        Returns:
            The path to the saved file
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        timestamp = int(time.time())
        file_path = f"{WAV_DIR}/{session_id}_{timestamp}.wav"

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    @staticmethod
    def create_conversation_with_audio(audio_path: str) -> List[Dict[str, Any]]:
        """
        Create a conversation structure with an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            A conversation structure that can be passed to the model
        """
        return [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]

    @staticmethod
    def get_previous_conversation(session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve previous conversation parts for a session.

        Args:
            session_id: The session ID to look for

        Returns:
            A conversation structure with all previous messages
        """
        # Get all messages from the history manager
        history_messages = ConversationHistoryManager.get_messages(session_id)

        if history_messages:
            return history_messages

        # If no history, check if there are audio files for this session
        audio_files = sorted(glob.glob(f"{WAV_DIR}/{session_id}_*.wav"))

        conversation = []

        # Add each audio file to the conversation
        for file_path in audio_files:
            conversation.append(
                {"role": "user", "content": [{"type": "audio", "audio": file_path}]}
            )

        return conversation

    @staticmethod
    def save_conversation_message(session_id: str, role: str, content: Any):
        """
        Save a message to the conversation history.

        Args:
            session_id: The session ID
            role: The role of the message sender ('user' or 'assistant')
            content: The content of the message
        """
        message = {"role": role, "content": content}
        ConversationHistoryManager.save_message(session_id, message)

    @staticmethod
    def process_conversation(conversation: List[Dict[str, Any]]) -> Generator:
        """
        Process a conversation with the model.

        Args:
            conversation: The conversation structure

        Returns:
            A generator that yields model responses
        """
        return inference(conversation)
