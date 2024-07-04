from pydantic import BaseModel, field_validator
from uuid import UUID
from PIL import Image
from io import BytesIO
import base64
from typing import List, Union

class Create2DRequestResponse(BaseModel):
    prompt: str
    request_id: Union[UUID, str, int]
    num: int
    images: List[str]

    @field_validator('images', mode='before')
    def validate_image(cls, v):
        for image_str in v:
            try:
                image_data = base64.b64decode(image_str)
                image = Image.open(BytesIO(image_data))
                image.verify()  # Verify that it is, indeed, an image
            except Exception as e:
                raise ValueError(f'Invalid image: {image_str}')
        return v

    class Config:
        arbitrary_types_allowed = True


class Create223DRequestResponse(BaseModel):
    # TODO: decide on the implementation
    request_id: Union[UUID, int, str]
    image_id: int