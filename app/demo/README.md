# Voice Chat API

This project implements an API for a voice-based chat system that uses AI to process audio input, generate responses, and convert those responses back to speech.

## Features

- Audio input processing
- Speech-to-Text using Qwen2.5-Omni model
- Text response generation
- Text-to-Speech conversion using Piper
- Streaming responses for real-time interaction
- Memory optimization for long conversations
- Context summarization for maintaining conversation history

## Requirements

- Python 3.8+
- PyTorch
- FastAPI
- Uvicorn
- NumPy
- Transformers
- Piper TTS

## Installation

1. Clone this repository
2. Make the startup script executable:
   ```bash
   chmod +x start_server.sh
   ```
3. Run the startup script:
   ```bash
   ./start_server.sh
   ```

The script will:
- Check for dependencies
- Install required Python packages
- Download the necessary models
- Start the server

## Usage

### Running the Server

```bash
./start_server.sh
```

Optional arguments:
- `--host`: Specify the host (default: 0.0.0.0)
- `--port`: Specify the port (default: 8000)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes (default: 1)

Example:
```bash
./start_server.sh --port 8080 --reload
```

### API Documentation

Once the server is running, you can access the API documentation at:
```
http://localhost:8000/docs
```

### Testing the API

You can use the included client script to test the API:

```bash
# Install additional requirements for the client
pip install pyaudio pydub

# Run the client
python client.py
```

## API Endpoints

### POST /chat

Send audio input and receive audio response.

Request body:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "audio",
          "audio": "<base64_encoded_audio_data>"
        }
      ]
    }
  ],
  "last_k_turns": 5,
  "summary": ""
}
```

### GET /health

Check if the API is running.

## Performance Optimization

The API includes several optimizations:

1. **Lazy Loading**: Models are loaded only when needed
2. **Memory Optimization**: Uses 4-bit quantization and streaming
3. **Conversation Window**: Only keeps recent messages in full detail
4. **Context Summarization**: Maintains conversation history without storing all audio data
5. **Streaming Response**: Processes and returns audio in chunks for faster response

## Customization

- Change the speech model by updating the `MODEL_NAME` constant
- Adjust the conversation window size with the `last_k_turns` parameter
- Change the TTS voice by updating the voice name in `init_models()`
