from fastapi import HTTPException, status, APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from uuid import UUID, uuid4
from typing import AsyncGenerator
from pathlib import Path
from typing import Union
from PIL import Image

from .models import Create3DRequestResponse, Create2DRequestResponse
from ml_pipeline.debug_prompt_to_img import generate_images


router = APIRouter()

async def _get_data_from_file(filepath: str) -> AsyncGenerator:
    '''
    This generator function needs to be used when parsing large chunks of Data.
    This must be used as the argument of StreamingResponse, in case the file is too large for FileResponse
    '''
    with open(filepath, mode="rb") as file_like:
        yield file_like.read()


@router.get("/request/2D/{prompt}", response_model=Create2DRequestResponse)
def gen_img_default(prompt: str) -> Create2DRequestResponse:
    '''
    endpoint "/request/2D/{prompt}" takes one path parameter:
    -prompt: user prompt
    '''

    num = 3
    
    if not num > 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The number must be an integer and bigger than 0."
            )
    
    request_id = uuid4()
    
    try:
        generate_images(prompt=prompt, n=num, request_id=str(request_id))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occured while generating images. Try again later."
            )
    imgs = [Image.open("data/images/" + str(request_id) + f"/{str(i)}.png") for i in range(num)]
    return Create2DRequestResponse(
        prompt=prompt,
        num=num,
        request_id=request_id,
        imgs=imgs
        )


@router.get("/request/2D/{prompt}/{num}", response_model=Create2DRequestResponse)
def gen_img(prompt: str, num: int) -> Create2DRequestResponse:
    '''
    endpoint "/request/2D/{prompt}{num}" takes two path parameters:
    -prompt: user prompt
    -num: number of generated images to choose from, default=3
    '''
    
    if not num > 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The number must be an integer and bigger than 0"
            )
    
    request_id = uuid4()
    
    try:
        generate_images(prompt=prompt, n=num, request_id=str(request_id))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occured while generating images. Try again later."
            )
    imgs = [Image.open("data/images/" + str(request_id) + f"/{str(i)}.png") for i in range(num)]
    return Create2DRequestResponse(
        prompt=prompt,
        num=num,
        request_id=request_id,
        imgs=imgs
        )

@router.get("/download/obj", response_class=StreamingResponse)
def download_obj(request_id: Union[UUID, str]) -> StreamingResponse:
    '''
    Endpoint "/download/obj" allows user to download a given .obj file by its id
    '''
    
    # something is... not quite right here.
    # The browser tab took a lot of my RAM at some point while doing the request
    # I'm more than sure that while testing we should not ever initiate the request
    # But after doing some risk tests I've got 200: success code, so it should work.
    
    # The issue is probably because the swagger UI which I used to test functionality
    # Tried to show me the file as text, which is enormous in itself.

    filepath = "data/3d_models/" + str(request_id) + "/refined.obj"
    
    if not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified file does not exist"
            )

    headers = {
        'Content-Disposition': f'attachment; filename="refined_mesh.obj"'
    }

    return StreamingResponse(_get_data_from_file(filepath), headers=headers, media_type="model/obj")
@router.get("/download/mtl/{request_id}/{num}", response_class=FileResponse)
def download_mtl(request_id: Union[UUID, str]) -> FileResponse:
    '''
    Endpoint "/download/obj" allows user to download a given .mtl file by its id
    '''
    
    filepath = "data/3d_models/" + str(request_id) + "/refined.mtl"
    
    if not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified file does not exist"
            )
    
    return FileResponse(path=filepath, filename="refined_mesh.mtl", media_type="model/mtl")

@router.get("/download/png/{request_id}/{num}", response_class=FileResponse)
def download_png(request_id: Union[UUID, str]) -> FileResponse:
    '''
    Endpoint "/download/obj" allows user to download a given .png file by its id
    '''
    
    filepath = "data/3d_models/" + str(request_id) + "/refined_albedo.png"
    
    if not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified file does not exist"
            )
    return FileResponse(path=filepath, filename="refined_albedo.png", media_type="image/png")