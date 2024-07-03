from pydantic import BaseModel
from uuid import UUID

class Create3DRequestResponse(BaseModel):
    prompt: str
    file_id: UUID

class Create2DRequestResponse(BaseModel):
    prompt: str
    request_id: UUID
    num: int

class Create223DRequestResponse(BaseModel):
    # TODO: decide on the implementation
    ...