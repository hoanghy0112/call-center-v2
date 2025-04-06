import wave
from piper.voice import PiperVoice
from uuid import uuid4

from app.constants import TTS_DIR

name = "en_US-hfc_female-medium"

model_path = f"./{TTS_DIR}/voices/{name}.onnx"
config_path = f"./{TTS_DIR}/voices/{name}.onnx.json"

voice = PiperVoice.load(model_path, config_path)


def generateSpeech(text):
    audio_buffer = bytearray()

    chunk_size = 1 * 22050 * 1 * 2 # duration * sample_rate * number_of_channels * sample_width

    for audio_bytes in voice.synthesize_stream_raw(text):
        audio_buffer.extend(audio_bytes)
        while len(audio_buffer) >= chunk_size:
            yield bytes(audio_buffer[:chunk_size])
            audio_buffer = audio_buffer[chunk_size:]
    
    if audio_buffer:
        yield bytes(audio_buffer)

    # filename = f"./{WAV_DIR}/output-{uuid4()}.wav"

    # with wave.open(filename, "wb") as wav_file:
    #     wav_file.setnchannels(1)
    #     wav_file.setsampwidth(2)
    #     wav_file.setframerate(SAMPLE_RATE)

    #     voice.synthesize(text, wav_file)
