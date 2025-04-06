import wave
from piper.voice import PiperVoice
from uuid import uuid4

from app.constants import TTS_DIR

name = "en_US-hfc_female-medium"

model_path = f"./{TTS_DIR}/voices/{name}.onnx"
config_path = f"./{TTS_DIR}/voices/{name}.onnx.json"

voice = PiperVoice.load(model_path, config_path)


def generateSpeech(text):
    for chunk in voice.synthesize_stream_raw(text):
        yield chunk

    # filename = f"./{WAV_DIR}/output-{uuid4()}.wav"

    # with wave.open(filename, "wb") as wav_file:
    #     wav_file.setnchannels(1)
    #     wav_file.setsampwidth(2)
    #     wav_file.setframerate(SAMPLE_RATE)

    #     voice.synthesize(text, wav_file)
