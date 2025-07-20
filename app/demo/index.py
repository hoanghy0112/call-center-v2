from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import torch
import numpy as np
import wave
import io
import time
from typing import List, Dict, Any
from pydantic import BaseModel
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread
from piper.voice import PiperVoice
from contextlib import asynccontextmanager

# Initialize FastAPI app
app = FastAPI()

# Constants
SAMPLE_RATE = 48000
CHUNK_SIZE = SAMPLE_RATE * 2  # 2 seconds worth of audio data
SILENCE_THRESHOLD = 0.01

MODEL_NAME = "Qwen2.5-Omni-3B-Q4_K_M.gguf"

# Model configurations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Initialize models (lazy loading)
processor = None
model = None
tokenizer = None
streamer = None
tts_voice = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_models():
    global processor, model, tokenizer, streamer, tts_voice

    if processor is None:
        # Load STT model
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        # Load TTS model
        name = "en_US-hfc_female-medium"
        model_path = f"./text_to_speech/voices/{name}.onnx"
        config_path = f"./text_to_speech/voices/{name}.onnx.json"
        tts_voice = PiperVoice.load(model_path, config_path)


class AudioMessage(BaseModel):
    role: str
    content: List[Dict[str, Any]]
    timestamp: float = time.time()  # Add timestamp for managing conversation window


class Conversation(BaseModel):
    messages: List[AudioMessage]
    summary: str = ""  # Store conversation summary
    last_k_turns: int = 5  # Number of recent turns to keep in active memory


def process_audio_stream(audio_data: bytes) -> np.ndarray:
    """Process raw audio data into the format expected by the model."""
    with io.BytesIO(audio_data) as buf:
        with wave.open(buf, "rb") as wav:
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    return audio


def summarize_conversation(messages: List[Dict]) -> str:
    """Create a brief summary of older conversation turns."""
    # Create a system message to summarize
    summary_prompt = "Summarize the key points of this conversation: \n"
    for msg in messages:
        if msg["role"] == "assistant":
            summary_prompt += f"Assistant: {msg.get('text', '')}\n"
        elif msg["role"] == "user":
            # For user messages, we only include transcribed text if available
            for content in msg["content"]:
                if content.get("type") == "text":
                    summary_prompt += f"User: {content['text']}\n"

    # Use the model to generate a summary
    summary_inputs = processor(text=summary_prompt, return_tensors="pt", padding=True)
    summary_inputs = {k: v.to(device) for k, v in summary_inputs.items()}

    with torch.no_grad():
        summary = model.generate(
            **summary_inputs,
            max_length=200,
            num_return_sequences=1,
        )
        summary_text = processor.tokenizer.decode(summary[0], skip_special_tokens=True)

    return summary_text


def generate_text_response(conversation: List[Dict], processor, model, streamer):
    """Generate text response from audio input using the model."""
    start_time = time.time()

    # Get the window size
    last_k_turns = conversation[-1].get("last_k_turns", 5)

    # Split conversation into recent and historical parts
    recent_messages = conversation[-last_k_turns:]
    historical_messages = (
        conversation[:-last_k_turns] if len(conversation) > last_k_turns else []
    )

    # Get or generate summary for historical context
    summary = conversation[-1].get("summary", "")
    if historical_messages and not summary:
        summary = summarize_conversation(historical_messages)

    # Create context-aware conversation
    if summary:
        context_messages = [
            {"role": "system", "content": f"Previous conversation context: {summary}"},
            *recent_messages,
        ]
    else:
        context_messages = recent_messages

    # Process conversation with context
    text = processor.apply_chat_template(
        context_messages, add_generation_prompt=True, tokenize=False
    )

    # Process only recent audio messages
    audios = []
    for message in recent_messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(ele["audio"])

    # Prepare inputs
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    # Generate response
    generation_kwargs = {
        **inputs,
        "max_length": 4024,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return streamer


def text_to_speech_stream(text: str):
    """Convert text to speech and stream the audio data."""
    audio_buffer = bytearray()
    chunk_size = 1 * 22050 * 1 * 2  # duration * sample_rate * channels * sample_width

    for audio_bytes in tts_voice.synthesize_stream_raw(text):
        audio_buffer.extend(audio_bytes)
        while len(audio_buffer) >= chunk_size:
            yield bytes(audio_buffer[:chunk_size])
            audio_buffer = audio_buffer[chunk_size:]

    if audio_buffer:
        yield bytes(audio_buffer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield


@app.post("/chat")
async def chat_endpoint(conversation: Conversation):
    """Main chat endpoint that handles audio input and generates audio response."""
    try:
        # Initialize models if not already done
        init_models()

        # Convert the Pydantic model to a list of dictionaries
        conv_list = [msg.model_dump() for msg in conversation.messages]

        # Add conversation metadata
        for msg in conv_list:
            msg["last_k_turns"] = conversation.last_k_turns
            msg["summary"] = conversation.summary

        # Generate text response
        text_stream = generate_text_response(conv_list, processor, model, streamer)

        # Convert text response to audio stream
        def response_generator():
            response_text = ""
            for chunk in text_stream:
                response_text += chunk
                if "." in chunk:  # Stream sentence by sentence
                    sentences = response_text.split(".")
                    response_text = sentences[-1]
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            yield from text_to_speech_stream(sentence + ".")

            # Process any remaining text
            if response_text.strip():
                yield from text_to_speech_stream(response_text)

        return StreamingResponse(
            response_generator(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
