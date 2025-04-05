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
import os


from app.api.main import api_router
from app.core.config import settings
from app.utils.remove_noise import remove_noise_from_bytearray

SAMPLE_RATE = 48000  # Audio sample rate in Hz
FRAME_DURATION = 30  # Frame duration in ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000) * 2
SILENCE_THRESHOLD = 2.0  # 1 second of silence

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

if not os.path.isdir("./wav_audio"):
    os.mkdir("./wav_audio")


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

    while True:
        try:
            data = await websocket.receive_bytes()
            data = remove_noise_from_bytearray(data, fs=SAMPLE_RATE)
            temp_buffer.extend(data)
        except:
            return

        has_voice = False
        count_void = 0
        total = len(temp_buffer) / FRAME_SIZE
        while len(temp_buffer) >= FRAME_SIZE:
            frame = temp_buffer[:FRAME_SIZE]
            temp_buffer = temp_buffer[FRAME_SIZE:]
            if not vad.is_speech(frame, SAMPLE_RATE):
                # has_voice = True
                count_void += 1

        now = time.time()

        ratio = count_void / total
        print("Voice ratio: ", ratio, now, sep=" - ")
        # print("void")
        if ratio < 0.5:
            last_voice_time = now
            is_saved = False
            audio_buffer.extend(data)
            print("speaking....")
            # print(uuid.uuid4())

        if now - last_voice_time > SILENCE_THRESHOLD and is_saved == False:
            print("now: ", now)
            print("last_voice_time: ", last_voice_time)
            is_saved = True

            print("Start...............................................")
            start_time = time.time()

            filename = f"./wav_audio/call_{index}_audio.wav"
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)  # mono audio
                wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_buffer)

            fileAudioData = librosa.load(
                f"{filename}", sr=processor.feature_extractor.sampling_rate
            )[0]

            elapsed_time = time.time() - start_time
            print("Elapsed time: ", elapsed_time)

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
    stereo_samples = stereo_samples.reshape(-1, 2).T

    # Average the two channels to get mono samples
    mono_samples = np.mean(stereo_samples, axis=0).astype(np.float16)

    return mono_samples / 32768.0
