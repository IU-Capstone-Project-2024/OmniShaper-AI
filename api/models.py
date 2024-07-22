from pydantic import BaseModel, field_validator
from uuid import UUID
from PIL import Image
from io import BytesIO
import base64
from typing import List, Union
# import magic

# class Create2DRequestResponse(BaseModel):
#     prompt: str
#     request_id: Union[UUID, str, int]
#     num: int
#     images: List[str]

#     @field_validator('images', mode='before')
#     def validate_image(cls, v):
#         for image_str in v:
#             try:
#                 image_data = base64.b64decode(image_str)
#                 image = Image.open(BytesIO(image_data))
#                 image.verify()  # Verify that it is, indeed, an image
#             except Exception as e:
#                 raise ValueError(f'Invalid image: {image_str}')
#         return v

#     class Config:
#         arbitrary_types_allowed = True

class Create2DRequestResponse(BaseModel):
    job_id: Union[UUID, str, int]
    request_id: Union[UUID, str, int]

class Create223DRequestResponse(BaseModel):
    job_id: Union[UUID, str, int]
    request_id: Union[UUID, str, int]
    image_id: int

# class Create223DRequestResponse(BaseModel):
#     request_id: Union[UUID, int, str]
#     image_id: int
#     obj_mtl_png: List[str]

    # @field_validator("obj_mtl_png")
    # def validate_obj_ntl(cls, v):
    #     allowed_mimetypes = {'model/obj', 'model/mtl', "image/png"}
    #     mime = magic.Magic(mime=True)
    #     for file_like in v:
    #         content = base64.b64decode(file_like)
    #         file_type = mime.from_buffer(content)
    #         if file_type not in allowed_mimetypes:
    #             raise ValueError("invalid file type.")
    #     return v

    # class Config:
    #     arbitrary_types_allowed = True
    
    # Right now the validation doesn't work as expected, but... We don't really need it