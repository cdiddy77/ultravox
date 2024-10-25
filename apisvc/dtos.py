from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from apisvc.reading import ReadingStatus, ReadingTaskState


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


class TaskStatusResponse(BaseModel):
    status: ReadingTaskState
