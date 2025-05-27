import time
import transformers
import numpy as np
from threading import Thread

from app.constants import (
    WAV_DIR,
    SAMPLE_RATE,
    FRAME_SIZE,
    SILENCE_THRESHOLD,
)

from app.utils.elapsed_decorator import timing_decorator

turns = [
    {
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people.",
    },
]

pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b", trust_remote_code=True
)


@timing_decorator
def inference(conversation):
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(ele["audio"])

    yield pipe(
        {"audios": audios, "turns": turns, "sampling_rate": SAMPLE_RATE}, max_new_tokens=30
    )
