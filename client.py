#!/usr/bin/env python3

import requests
import json
import wave
import pyaudio
import base64
import argparse
import sys
import os
from pydub import AudioSegment
from pydub.playback import play
import io
import numpy as np
import time


def record_audio(duration=5, sample_rate=48000):
    """Record audio from the microphone"""
    print(f"Recording for {duration} seconds...")

    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()

    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    frames = []

    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert to WAV
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    buffer.seek(0)
    return buffer.read()


def play_audio_stream(response):
    """Play the streaming audio response"""
    # Set up PyAudio
    p = pyaudio.PyAudio()

    # Open a stream to play the audio
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=22050,  # Match the rate used in the server's TTS
        output=True,
    )

    # Read the streaming response and play the audio
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            stream.write(chunk)

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()


def send_audio_to_api(audio_data, api_url, conversation_history=None):
    """Send audio data to the API and play the response"""
    # Encode audio data as base64
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # Create the request payload
    if conversation_history is None:
        conversation_history = []

    # Add the new message
    conversation_history.append(
        {"role": "user", "content": [{"type": "audio", "audio": audio_base64}]}
    )

    payload = {
        "messages": conversation_history,
        "last_k_turns": 5,  # Keep last 5 turns in detail
    }

    # Send the request
    print("Sending request to API...")
    response = requests.post(
        api_url, json=payload, stream=True, headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return conversation_history

    # Play the audio response
    print("Playing response...")
    play_audio_stream(response)

    # Add the assistant's response to the conversation history
    conversation_history.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "[Voice response]"}],  # Placeholder
        }
    )

    return conversation_history


def main():
    parser = argparse.ArgumentParser(description="Audio Chat Client")
    parser.add_argument("--url", default="http://localhost:8000/chat", help="API URL")
    parser.add_argument(
        "--duration", type=int, default=5, help="Recording duration in seconds"
    )
    args = parser.parse_args()

    conversation_history = []

    try:
        while True:
            # Record audio
            input("Press Enter to start recording...")
            audio_data = record_audio(duration=args.duration)

            # Send to API and update conversation history
            conversation_history = send_audio_to_api(
                audio_data, args.url, conversation_history
            )

            print("\nConversation continues. Press Ctrl+C to exit.")

    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
