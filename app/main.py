from fastapi import FastAPI, WebSocket
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
import time
import wave
import webrtcvad
import librosa
import numpy as np
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import io
import soundfile as sf
import uuid

from app.api.main import api_router
from app.core.config import settings

SAMPLE_RATE = 48000  # Audio sample rate in Hz
FRAME_DURATION = 30  # Frame duration in ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000) * 2
SILENCE_THRESHOLD = 1.0  # 1 second of silence

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")


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
    is_saved = False

    while True:
        try:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            temp_buffer.extend(data)
        except:
            return

        while len(temp_buffer) >= FRAME_SIZE:
            frame = temp_buffer[:FRAME_SIZE]
            temp_buffer = temp_buffer[FRAME_SIZE:]
            if vad.is_speech(frame, SAMPLE_RATE):
                last_voice_time = time.time()
                is_saved = False

        if time.time() - last_voice_time > SILENCE_THRESHOLD and is_saved == False:
            is_saved = True

            print("Start...")
            start_time = time.time()

            filename = f"call_{index}_audio.wav"
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)  # mono audio
                wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_buffer)

            fileAudioData = librosa.load(
                f"./{filename}", sr=processor.feature_extractor.sampling_rate
            )[0]
            # print(f"Audio saved to {filename} after 1s of silence.")

            elapsed_time = time.time() - start_time
            print("Elapsed time: ", elapsed_time)

            bufferAudioData = get_spectrogram(
                audio_buffer, processor.feature_extractor.sampling_rate
            )

            arr = np.array(fileAudioData)
            np.savetxt("fileAudioData.txt", arr)

            arr2 = np.array(bufferAudioData)
            np.savetxt("bufferAudioData.txt", arr2)

            print("fileAudioData: ", arr.shape)
            print("bufferAudioData: ", arr2.shape)

            await websocket.send_text(f"Audio saved to {filename} after 1s of silence.")

            index += 1
            audio_buffer = bytearray()
            temp_buffer = bytearray()
            vad = webrtcvad.Vad(3)
            last_voice_time = time.time()


def get_spectrogram(buffer, target_sr):
    # y, sr = sf.read(buffer, dtype='int16')

    # if sr != target_sr:
    #     y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # audio_data = np.frombuffer(buffer, dtype=np.int16)
    # audio_data = librosa.util.buf_to_float(audio_data)
    # audio_data /= 32768.0

    # spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE)

    stereo_samples = np.frombuffer(buffer, dtype=np.int16)

    # Reshape to separate channels: (2, N)
    stereo_samples = stereo_samples.reshape(-1, 3).T

    # Average the two channels to get mono samples
    mono_samples = np.mean(stereo_samples, axis=0).astype(np.float16)

    return mono_samples / 32768.0
