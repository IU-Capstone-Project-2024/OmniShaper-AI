from pydantic import BaseModel
from uuid import UUID

class CreateRequestResponse(BaseModel):
    prompt: str
    file_id: UUID