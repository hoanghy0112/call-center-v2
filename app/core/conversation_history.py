import json
import os
import time
from typing import Dict, Any

from app.constants import WAV_DIR


class ConversationHistoryManager:
    """Manages the storage and retrieval of conversation history."""

    HISTORY_DIR = f"{WAV_DIR}/history"

    @classmethod
    def ensure_history_dir(cls):
        """Ensure the history directory exists."""
        if not os.path.exists(cls.HISTORY_DIR):
            os.makedirs(cls.HISTORY_DIR, exist_ok=True)

    @classmethod
    def save_message(cls, session_id: str, message: Dict[str, Any]):
        """
        Save a message to the conversation history.

        Args:
            session_id: The session ID
            message: The message to save
        """
        cls.ensure_history_dir()

        timestamp = int(time.time())
        filename = f"{cls.HISTORY_DIR}/{session_id}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(message, f)

    @classmethod
    def get_messages(cls, session_id: str):
        """
        Get all messages for a session.

        Args:
            session_id: The session ID

        Returns:
            List of messages in chronological order
        """
        cls.ensure_history_dir()

        # Get all files for this session
        import glob

        files = sorted(glob.glob(f"{cls.HISTORY_DIR}/{session_id}_*.json"))

        messages = []
        for file in files:
            try:
                with open(file, "r") as f:
                    message = json.load(f)
                    messages.append(message)
            except Exception as e:
                print(f"Error reading message file {file}: {str(e)}")

        return messages
