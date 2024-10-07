from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, File, Request, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import tempfile
from apisvc.dtos import UploadAudioResponse
import structlog

from apisvc.stts_task import (
    process_audio,
)


log: structlog.stdlib.BoundLogger = structlog.get_logger()


# args = simple_parsing.parse(Config)
sse_queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await setup_elevenlabs_websocket()

    yield
    # await close_elevenlabs_websocket()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/status")
async def status():
    return {"status": "ok"}


@app.post("/upload-audio/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    prompt: str = Form("<|audio|>respond as a roma gypsy"),
    max_new_tokens: int = Form(50),
    temperature: float = Form(0.0),
) -> UploadAudioResponse:
    log.info(
        "Received audio upload",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    # Save the uploaded audio file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    background_tasks.add_task(
        process_audio, tmp_path, max_new_tokens, temperature, prompt, sse_queue
    )
    return UploadAudioResponse(status="processing")


@app.get("/response-events")
async def sse_endpoint(request: Request):
    log.info("Received request for SSE")

    async def event_stream():
        while True:
            if await request.is_disconnected():
                log.info("SSE Client disconnected")
                break

            # Wait for new data in the queue
            data = await sse_queue.get()
            # Send data as a server-sent event
            yield f"data: {data}\n\n"

    # Return a streaming response with content type as text/event-stream
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7799)
