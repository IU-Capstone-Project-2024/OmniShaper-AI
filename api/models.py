from pydantic import BaseModel
from uuid import UUID, uuid4

class CreateRequestResponse(BaseModel):
    prompt: str
    file_id: UUID