from typing import Generator
import structlog
import asyncio
import websockets
import json
import base64
from apisvc.config import InferenceConfig, get_config
from ultravox.data import datasets
from ultravox.inference import ultravox_infer, base as infer_base

inference_singleton: ultravox_infer.UltravoxInference | None = None
log: structlog.stdlib.BoundLogger = structlog.get_logger()

ELEVENLABS_API_KEY = get_config("ELEVENLABS_API_KEY")
SEER_MORGANA_VOICE_ID = "7NsaqHdLuKNFvEfjpUno"


def init_inference(args):
    global inference_singleton
    if not inference_singleton:
        inference_singleton = ultravox_infer.UltravoxInference(
            args.model_path,
            device=args.device,
            data_type=args.data_type,
            conversation_mode=True,
        )


def get_inference():
    global inference_singleton
    return inference_singleton


args = InferenceConfig()

init_inference(args)


async def stream(audio_stream, queue: asyncio.Queue):
    log.info("Started streaming audio")
    async for chunk in audio_stream:
        if chunk:
            await queue.put(chunk)
            log.info("Received audio chunk")


async def text_to_speech_input_streaming(voice_id, text_iterator, queue: asyncio.Queue):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2_5"

    async with websockets.connect(uri) as websocket:
        log.info("Connected to websocket")
        await websocket.send(
            json.dumps(
                {
                    "text": " ",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                    "xi_api_key": ELEVENLABS_API_KEY,
                }
            )
        )
        log.info("Sent initial message")

        async def listen():
            """Listen to the websocket for audio data and stream it."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    log.info("Received message")
                    if data.get("audio"):
                        yield data["audio"]
                    elif data.get("isFinal"):
                        break
                except websockets.exceptions.ConnectionClosed:
                    log.info("Connection closed")
                    break

        listen_task = asyncio.create_task(stream(listen(), queue))

        # async for text in text_chunker(text_iterator):
        async for text in text_iterator:
            await websocket.send(json.dumps({"text": text}))

        await websocket.send(json.dumps({"text": ""}))

        await listen_task


async def process_audio(
    tmp_path: str,
    max_new_tokens: int,
    temperature: float,
    prompt: str,
    queue: asyncio.Queue,
):
    log.info(
        "Processing audio",
        tmp_path=tmp_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt=prompt,
    )
    # Load the audio file and create a VoiceSample
    sample = datasets.VoiceSample.from_prompt_and_file(prompt, tmp_path)

    inference = get_inference()
    if not inference:
        raise ValueError("Inference object not initialized")
    # Perform inference
    output = inference.infer_stream(
        sample,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    async def text_chunk_iterator():
        for chunk in output:
            if isinstance(chunk, infer_base.InferenceChunk):
                yield chunk.text

    await text_to_speech_input_streaming(
        SEER_MORGANA_VOICE_ID, text_chunk_iterator(), queue
    )
