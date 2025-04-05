WAV_DIR = "wav_audio"
TTS_DIR = "text_to_speech"

SAMPLE_RATE = 48000  # Audio sample rate in Hz
FRAME_DURATION = 30  # Frame duration in ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000) * 2
SILENCE_THRESHOLD = 2.0  # 1 second of silence
