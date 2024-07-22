from MicroDreamer.pipeline import ImgTo3dPipeline
import os
import warnings
from api.api_utils import convert_files
from typing import List

def generate_3d_model(request_id, image_id) -> List[str]:
    warnings.filterwarnings("ignore")

    try:
        request_id = str(request_id)
        image_id = str(image_id)
    except Exception as e:
        pass

    filename = f'{request_id}_{image_id}'

    img_to_3d = ImgTo3dPipeline()

    current_dir = os.getcwd()

    # Define the target directory
    target_dir = 'MicroDreamer'

    # Change the directory only if the current directory is not the target directory
    if os.path.basename(current_dir) != target_dir:
        os.chdir(target_dir)

    img_to_3d(f'data/images/{request_id}/{image_id}.png', 512)

    current_dir = os.getcwd()
    if os.path.basename(current_dir) == target_dir:
        os.chdir('..')
    print(os.path.basename(os.getcwd()))

    return convert_files(request_id=request_id, image_id=image_id)