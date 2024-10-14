from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, File, Request, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import tempfile
from apisvc.dtos import ResetConversationResponse, UploadAudioResponse
import structlog

from apisvc.stts_task import process_audio, update_conversation


log: structlog.stdlib.BoundLogger = structlog.get_logger()

# processors = [
#     structlog.contextvars.merge_contextvars,
#     # structlog.processors.add_log_level,
#     # structlog.dev.set_exc_info,
#     structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
# ]
# structlog.configure(processors=processors)

# args = simple_parsing.parse(Config)
sse_queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await setup_elevenlabs_websocket()

    yield

    log.info("Shutting down")
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
    if "<|audio|>" not in prompt:
        prompt = "<|audio|>" + prompt
    # Save the uploaded audio file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    background_tasks.add_task(
        process_audio, tmp_path, max_new_tokens, temperature, prompt, sse_queue
    )
    return UploadAudioResponse(status="processing")


@app.post("/reset-conversation")
async def reset_conversation(
    background_tasks: BackgroundTasks,
) -> ResetConversationResponse:
    log.info("received reset conversation request")
    background_tasks.add_task(update_conversation, [])
    return ResetConversationResponse(status="processing")


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
            if data == "complete":
                yield "event: close\ndata: audio processing complete\n\n"
                break
            elif data == "error":
                yield "event: error\ndata: audio processing error\n\n"
                break
            # Send data as a server-sent event
            yield f"data: {data}\n\n"

    # Return a streaming response with content type as text/event-stream
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7799)
