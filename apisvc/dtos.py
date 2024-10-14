from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field


class UploadAudioResponse(BaseModel):
    status: Literal["pending", "processing", "completed", "failed"]


class ResetConversationResponse(BaseModel):
    status: Literal["pending", "processing", "completed", "failed"]
