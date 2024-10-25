import asyncio
import base64
from io import BytesIO
from typing import Literal, Optional
import uuid
from elevenlabs import AsyncElevenLabs, ElevenLabs, VoiceSettings
import openai
from pydantic import BaseModel, Field
import structlog

from apisvc.config import get_config

log: structlog.stdlib.BoundLogger = structlog.get_logger()

ELEVENLABS_API_KEY = get_config("ELEVENLABS_API_KEY")
SEER_MORGANA_VOICE_ID = "7NsaqHdLuKNFvEfjpUno"

ReadingStatus = Literal[
    "no_cards",
    "verifying_cards",
    "requesting_reading",
    "error",
    "reading_tts_requested",
    "reading_tts_complete",
]


class TarotCard(BaseModel):
    name: str = Field(..., description="The tarot card name.")
    description: str = Field(..., description="The description of the tarot card.")


class TarotCardHand(BaseModel):
    cards: list[TarotCard] = Field(
        ...,
        description="The cards in the tarot card hand. This can be empty if the image does not contain any tarot cards",
    )


class ReadingTaskState(BaseModel):
    status: ReadingStatus
    task_id: Optional[str] = None
    # if status == "error"
    error_message: Optional[str] = None
    # if status == "verifying_cards"
    # or status == "requesting_reading"
    # or status == 'reading_tts_requested'
    # or status == 'reading_tts_complete'
    hand: Optional[TarotCardHand] = None

    # if status == 'reading_tts_complete'
    # base64 encoded audio file
    reading: Optional[str] = None


reading_task_states: dict[str, ReadingTaskState] = {}


def get_reading_task_state(task_id: str) -> ReadingTaskState | None:
    return reading_task_states.get(task_id, None)


def set_reading_task_state(task_id: str, state: ReadingTaskState):
    log.info("Setting task state", task_id=task_id, state=state.status, hand=state.hand)
    reading_task_states[task_id] = state


async def process_tarot_cards(task_id: str):
    state = get_reading_task_state(task_id)
    client = openai.AsyncOpenAI(api_key=get_config("OPENAI_API_KEY"))

    if not state or not state.hand:
        log.error(
            "Task state not found or hand not found", task_id=task_id, state=state
        )
        set_reading_task_state(
            task_id=task_id,
            state=ReadingTaskState(
                status="error",
                task_id=task_id,
                error_message=f"Task state not found or hand not found: {state}",
            ),
        )
        return

    cards_text = "\n".join([card.name for card in state.hand.cards if state.hand or []])
    # generate a reading
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a slightly nutty, very wise and humorous roma gypsy",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
Here are a set of tarot cards. Please provide an entertaining reading for me. 
Do not include any formatting.\n
"""
                            + cards_text,
                        },
                    ],
                },
            ],
            max_tokens=300,
        )

        reading_text = response.choices[0].message.content
    except Exception as e:
        log.error("Error calling OpenAI", error=str(e))
        set_reading_task_state(
            task_id=task_id,
            state=ReadingTaskState(
                status="error",
                task_id=task_id,
                error_message=f"Error calling OpenAI: {e}",
            ),
        )
        return

    set_reading_task_state(
        task_id=task_id,
        state=ReadingTaskState(
            status="reading_tts_requested",
            task_id=task_id,
            hand=state.hand,
        ),
    )

    # tts the reading
    try:
        client = ElevenLabs(
            api_key=ELEVENLABS_API_KEY,
        )
        response = client.text_to_speech.convert(
            voice_id=SEER_MORGANA_VOICE_ID,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=reading_text or "",
            voice_settings=VoiceSettings(
                stability=0.1,
                similarity_boost=0.3,
                style=0.2,
            ),
        )
        # store the audio in the task state
        audio_stream = BytesIO()

        # Write each chunk of audio data to the stream
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)

        # Reset stream position to the beginning
        audio_stream.seek(0)

        save_file_path = f"{uuid.uuid4()}.mp3"
        # Writing the audio stream to the file

        audio_data = audio_stream.getvalue()
        with open(save_file_path, "wb") as f:
            f.write(audio_data)

        audio_base64_str = base64.b64encode(audio_data).decode("utf-8")

        set_reading_task_state(
            task_id=task_id,
            state=ReadingTaskState(
                status="reading_tts_complete",
                task_id=task_id,
                hand=state.hand,
                reading=audio_base64_str,
            ),
        )
        # Update the task state to completed
        # task_states[task_id]["status"] = "completed"
    except Exception as e:
        log.error("Error calling TTS", error=str(e))
        set_reading_task_state(
            task_id=task_id,
            state=ReadingTaskState(
                status="error",
                task_id=task_id,
                error_message=f"Error calling TTS: {e}",
            ),
        )
