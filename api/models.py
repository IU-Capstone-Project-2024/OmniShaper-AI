from pydantic import BaseModel
from uuid import UUID
from PIL import Image
from typing import List

class Create2DRequestResponse(BaseModel):
    prompt: str
    request_id: UUID
    num: int
    imgs: List[(Image.Image, int)]

class Create223DRequestResponse(BaseModel):
    # TODO: decide on the implementation
    request_id: UUID
    image_id: int