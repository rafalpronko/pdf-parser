"""Error response models."""

from datetime import datetime

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str
    timestamp: datetime
    request_id: str
