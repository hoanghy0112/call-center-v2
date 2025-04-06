import uuid
from fastapi import FastAPI, WebSocket
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
import time
import wave
import webrtcvad
import librosa
from transformers import AutoProcessor
import os
from fastapi.staticfiles import StaticFiles
import threading


from app.api.main import api_router
from app.core.config import settings
from app.utils.remove_noise import remove_noise_from_bytearray
from app.audio_to_text import inference
from app.text_to_speech.main import generateSpeech
from app.constants import (
    WAV_DIR,
    SAMPLE_RATE,
    FRAME_SIZE,
    SILENCE_THRESHOLD,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

if not os.path.isdir(f"./{WAV_DIR}"):
    os.mkdir(f"./{WAV_DIR}")


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"/docs/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="")


@app.websocket("/calls/{call_id}/web_socket")
async def join_call_room(websocket: WebSocket, call_id: str):
    await websocket.accept()

    index = 0
    audio_buffer = bytearray()
    temp_buffer = bytearray()
    vad = webrtcvad.Vad(3)
    last_voice_time = time.time()
    is_saved = True

    conversation = []

    model_session_id = uuid.uuid4()

    async def send_response(session_id):
        for response, inference_time in inference(conversation):
            conversation.append(
                {
                    "role": "assistant",
                    "content": response,
                },
            )

            speech = generateSpeech(response)

            for chunk in speech:
                if session_id != model_session_id:
                    return
                print("Sending audio chunk...........................")
                await websocket.send_bytes(chunk)

            print("Inference time: ", inference_time)

    while True:
        try:
            data = await websocket.receive_bytes()
            data = remove_noise_from_bytearray(data, fs=SAMPLE_RATE)
            temp_buffer.extend(data)
        except:
            return

        count_void = 0
        total = len(temp_buffer) / FRAME_SIZE
        while len(temp_buffer) >= FRAME_SIZE:
            frame = temp_buffer[:FRAME_SIZE]
            temp_buffer = temp_buffer[FRAME_SIZE:]
            if not vad.is_speech(frame, SAMPLE_RATE):
                count_void += 1

        now = time.time()

        ratio = count_void / total
        if ratio < 0.5:
            last_voice_time = now
            is_saved = False
            audio_buffer.extend(data)

        if now - last_voice_time > SILENCE_THRESHOLD and is_saved == False:
            is_saved = True

            filename = f"./{WAV_DIR}/call_{index}_audio.wav"
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)  # mono audio
                wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_buffer)

            fileAudioData = librosa.load(
                f"{filename}", sr=processor.feature_extractor.sampling_rate
            )[0]

            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": fileAudioData,
                        },
                    ],
                }
            )

            model_session_id = uuid.uuid4()

            thread = threading.Thread(
                target=send_response, kwargs={"session_id": model_session_id}
            )
            thread.start()

            index += 1
            audio_buffer = bytearray()
            temp_buffer = bytearray()
            vad = webrtcvad.Vad(3)
            last_voice_time = time.time()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
