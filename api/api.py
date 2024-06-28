from fastapi import HTTPException, status, APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from uuid import UUID, uuid4
from typing import AsyncGenerator
from pathlib import Path

from .models import CreateRequestResponse
from ml_pipeline.pipeline import promt_to_3D

router = APIRouter()

async def _get_data_from_file(filepath: str) -> AsyncGenerator:
    '''
    This generator function needs to be used when parsing large chunks of Data.
    This must be used as the argument of StreamingResponse, in case the file is too large for FileResponse
    '''
    with open(filepath, mode="rb") as file_like:
        yield file_like.read()


@router.get("/request", response_model=CreateRequestResponse)
def create_request(prompt: str) -> CreateRequestResponse:
    '''
    The endpint "/request/{prompt}" takes user input, generates file_id, initiates 3D generating process
    and returns assigned file_id with the used prompt.
    '''
    
    file_id = uuid4()

    try:
        promt_to_3D(promt=prompt, filename=file_id)
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Error occured while creating 3D object.")

    return CreateRequestResponse(file_id=file_id, prompt=prompt)

@router.get("/download/obj", response_class=StreamingResponse)
def download_obj(file_id: UUID | str) -> StreamingResponse:
    '''
    Endpoint "/download/obj" allows user to download a given .obj file by its id
    '''
    
    # something is... not quite right here.
    # The browser tab took a lot of my RAM at some point while doing the request
    # I'm more than sure that while testing we should not ever initiate the request
    # But after doing some risk tests I've got 200: success code, so it should work.
    
    # The issue is probably because the swagger UI which I used to test functionality
    # Tried to show me the file as text, which is enormous in itself.

    filepath = "data/3d_models/" + str(file_id) + "_mesh.obj"
    
    if not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified file does not exist"
            )

    headers = {
        'Content-Disposition': f'attachment; filename="{file_id}_mesh.obj"'
    }

    return StreamingResponse(_get_data_from_file(filepath), headers=headers, media_type="model/obj")
@router.get("/download/mtl", response_class=FileResponse)
def download_mtl(file_id: UUID| str) -> FileResponse:
    '''
    Endpoint "/download/obj" allows user to download a given .mtl file by its id
    '''
    
    filepath = "data/3d_models/" + str(file_id) + "_mesh.mtl"
    
    if not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified file does not exist"
            )
    
    return FileResponse(path=filepath, filename=f"{str(file_id)}_mesh.mtl", media_type="model/mtl")

@router.get("/download/png", response_class=FileResponse)
def download_png(file_id: UUID | str) -> FileResponse:
    '''
    Endpoint "/download/obj" allows user to download a given .png file by its id
    '''
    
    filepath = "data/3d_models/" + str(file_id) + "_mesh_albedo.png"
    
    if not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified file does not exist"
            )
    return FileResponse(path=filepath, filename=f"{str(file_id)}_mesh_albedo.png", media_type="image/png")