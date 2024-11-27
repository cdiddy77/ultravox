from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from apisvc.reading import ReadingStatus, ReadingTaskState, TarotCardHand


class UploadAudioResponse(BaseModel):
    status: Literal["pending", "processing", "completed", "failed"]


class ResetConversationRequest(BaseModel):
    system_message: str = Field(
        ..., description="The system message to reset the conversation to."
    )


class ResetConversationResponse(BaseModel):
    status: Literal["pending", "processing", "completed", "failed"]


class UploadImageResponse(BaseModel):
    status: ReadingStatus
    task_id: Optional[str] = None


class SpotCardsResponse(BaseModel):
    hand: Optional[TarotCardHand] = None
    hand_verified: bool


class TaskStatusResponse(BaseModel):
    status: ReadingTaskState


class MessageRequest(BaseModel):
    phone_number: str
    message: str
