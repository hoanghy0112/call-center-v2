from fastapi import FastAPI, WebSocket
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoProcessor
import os
from fastapi.staticfiles import StaticFiles


from app.api.main import api_router
from app.core.config import settings
from app.api.websockets.call_websocket import handle_join_call_room
from app.constants import WAV_DIR

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
    try:
        await handle_join_call_room(websocket, call_id)
    except:
        pass


app.mount("/", StaticFiles(directory="static", html=True), name="static")
