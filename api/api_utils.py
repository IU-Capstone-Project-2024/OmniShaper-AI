from io import BytesIO
from PIL import Image
from typing import Union
from uuid import UUID
import base64

def _image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to a byte array
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_byte_array = buffered.getvalue()

        # Encode the byte array using base64
        img_base64 = base64.b64encode(img_byte_array).decode('utf-8')
        return img_base64

def _file_to_base64(file_path):
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string

def convert_images(request_id: Union[UUID, str, int], image_id: int):
    images = [_image_to_base64(f"data/images/{str(request_id)}/{str(i)}.png") for i in range(image_id)]
    return images

def convert_files(request_id: Union[UUID, str, int], image_id: int):
    generated_filepath = f"/data/3d_models/{str(request_id)}/{str(image_id)}/"
    
    img_refined = generated_filepath + "refined_albedo.png"
    obj_refined = generated_filepath + "refined.obj"
    mtl_refined = generated_filepath + "refined.mtl"

    obj_mtl_png = [
        _file_to_base64(obj_refined),
        _file_to_base64(mtl_refined),
        _image_to_base64(img_refined)
    ]

    return obj_mtl_png