import base64
from contextlib import asynccontextmanager
import asyncio
import uuid
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    Form,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import openai
import uvicorn
import tempfile
from apisvc.config import get_config
from twilio.rest import Client
from apisvc.dtos import (
    MessageRequest,
    ResetConversationRequest,
    ResetConversationResponse,
    TaskStatusResponse,
    UploadAudioResponse,
    UploadImageResponse,
)
import structlog

from apisvc.reading import (
    ReadingTaskState,
    TarotCardHand,
    get_reading_task_state,
    process_tarot_cards,
    set_reading_task_state,
)
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
    request: ResetConversationRequest,
    background_tasks: BackgroundTasks,
) -> ResetConversationResponse:
    log.info("received reset conversation request", request=request)
    background_tasks.add_task(
        update_conversation,
        [
            {
                "role": "system",
                "content": request.system_message,
            }
        ],
    )
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


@app.post("/upload-image/")
async def upload_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    task_id: str = Form(""),
):
    # Save the uploaded image to a temporary file
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    #     tmp.write(await image.read())
    #     tmp_path = tmp.name

    # def encode_image(image_path):
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode('utf-8')
    encoded_image = base64.b64encode(image.file.read()).decode("utf-8")

    client = openai.AsyncOpenAI(api_key=get_config("OPENAI_API_KEY"))

    # Call OpenAI to determine whether there are any tarot cards in the image
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the image for tarot cards. Return all tarot cards found, or an empty list if there are none.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tarot_cards",
                    "schema": TarotCardHand.model_json_schema(),
                },
            },
            max_tokens=300,
        )
    except Exception as e:
        log.error("Error calling OpenAI", error=str(e))
        raise HTTPException(status_code=500, detail="Error calling OpenAI")

    hand: TarotCardHand = TarotCardHand.model_validate_json(
        response.choices[0].message.content or "{}"
    )

    log.info("Received response from OpenAI", hand=hand)

    # if the reading state is verifying, then we get the current state
    # and see if they are the same, in which case it is verified.
    if not hand or not hand.cards:
        if task_id:
            set_reading_task_state(
                task_id,
                ReadingTaskState(status="no_cards", task_id=task_id, hand=hand),
            )
        return UploadImageResponse(status="no_cards")
    elif task_id:
        current_state = get_reading_task_state(task_id)
        if current_state and current_state.hand:
            # create a set of the current hand names
            # create a set of the new hand names
            # if they are the same, then we are verified
            current_hand_names = set(
                [card.name for card in current_state.hand.cards if current_state.hand]
            )
            new_hand_names = set([card.name for card in hand.cards])
            log.info(
                "checking hand",
                current_hand_names=current_hand_names,
                new_hand_names=new_hand_names,
            )
            if current_hand_names == new_hand_names:
                set_reading_task_state(
                    task_id=task_id,
                    state=ReadingTaskState(
                        status="requesting_reading", task_id=task_id, hand=hand
                    ),
                )
                background_tasks.add_task(func=process_tarot_cards, task_id=task_id)
                return UploadImageResponse(status="requesting_reading", task_id=task_id)
            else:
                set_reading_task_state(
                    task_id=task_id,
                    state=ReadingTaskState(
                        status="verifying_cards", task_id=task_id, hand=hand
                    ),
                )
                return UploadImageResponse(status="verifying_cards", task_id=task_id)
        else:
            set_reading_task_state(
                task_id=task_id,
                state=ReadingTaskState(
                    status="verifying_cards", task_id=task_id, hand=hand
                ),
            )
            return UploadImageResponse(status="verifying_cards", task_id=task_id)
    else:
        # Create a unique task id
        task_id = str(uuid.uuid4())

        # Store the initial state of the task
        set_reading_task_state(
            task_id, ReadingTaskState(status="verifying_cards", hand=hand)
        )

        return UploadImageResponse(status="verifying_cards", task_id=task_id)

    # # Create a background task to process the identified tarot cards
    # background_tasks.add_task(process_tarot_cards, task_id, response["cards"])

    # return JSONResponse(
    #     status_code=200, content={"status": "verifying cards", "task_id": task_id}
    # )


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    task_state = get_reading_task_state(task_id)
    if not task_state:
        return TaskStatusResponse(
            status=ReadingTaskState(
                status="error", error_message="Task not found", task_id=task_id
            )
        )

    return TaskStatusResponse(status=task_state)


@app.post("/send-message/")
async def send_message(request: MessageRequest):
    try:
        client = Client(
            get_config("TWILIO_ACCOUNT_SID"), get_config("TWILIO_AUTH_TOKEN")
        )
        message = client.messages.create(
            body=request.message,
            from_=get_config("TWILIO_PHONE_NUMBER"),
            to=request.phone_number,
        )
        return {"status": "success", "sid": message.sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7799)
