
import cv2
import time
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor

import os
import torch
from PIL import Image
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from einops import rearrange
import numpy as np
import argparse
import subprocess

def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    return ndarr


def save_image_to_disk(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


weight_dtype = torch.float16

_TITLE = '''Wonder3D: Single Image to 3D using Cross-Domain Diffusion'''
_DESCRIPTION = '''
<div>
Generate consistent multi-view normals maps and color images.
<a style="display:inline-block; margin-left: .5em" href='https://github.com/xxlong0/Wonder3D/'><img src='https://img.shields.io/github/stars/xxlong0/Wonder3D?style=social' /></a>
</div>
<div>
The demo does not include the mesh reconstruction part, please visit <a href="https://github.com/xxlong0/Wonder3D/">our github repo</a> to get a textured mesh.
</div>
'''
_GPU_ID = 0

if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image


def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{_GPU_ID}")
    predictor = SamPredictor(sam)
    return predictor


def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Background Removal" in chk_group
        rescale = "Rescale" in chk_group
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = image_arr[y:y + h,
                                                                                                 x:x + w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)


def load_wonder3d_pipeline(cfg):
    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path,
                                                                  subfolder="image_encoder", revision=cfg.revision)
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path,
                                                           subfolder="feature_extractor", revision=cfg.revision)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    unet = UNetMV2DConditionModel.from_pretrained_2d(cfg.pretrained_unet_path, subfolder="unet", revision=cfg.revision,
                                                     **cfg.unet_from_pretrained_kwargs)
    unet.enable_xformers_memory_efficient_attention()

    # Move text_encode and vae to gpu and cast to weight_dtype
    image_encoder.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    unet.to(dtype=weight_dtype)

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, vae=vae, unet=unet, safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler"),
        **cfg.pipe_kwargs
    )

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    # sys.main_lock = threading.Lock()
    return pipeline


from mvdiffusion.data.single_image_dataset import SingleImageDataset


def prepare_data(single_image, crop_size):
    dataset = SingleImageDataset(
        root_dir='',
        num_views=6,
        img_wh=[256, 256],
        bg_color='white',
        crop_size=crop_size,
        single_image=single_image
    )
    return dataset[0]


def run_pipeline(pipeline, cfg, single_image, guidance_scale, steps, seed, crop_size, chk_group=None, save_to=None):
    import pdb
    # pdb.set_trace()

    if chk_group is not None:
        write_image = "Write Results" in chk_group

    batch = prepare_data(single_image, crop_size)

    pipeline.set_progress_bar_config(disable=False)
    seed = int(seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(weight_dtype)

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(weight_dtype)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(
        weight_dtype)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(weight_dtype)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
    # (B*Nv, Nce)
    # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

    out = pipeline(
        imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
        num_inference_steps=steps,
        output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]
    num_views = 6
    if write_image:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        if save_to is not None:
            cur_dir = os.path.join("./outputs", save_to)
        else:
            cur_dir = os.path.join("./outputs", f"cropsize-{crop_size}-cfg{guidance_scale:.1f}")

        scene = 'scene'
        scene_dir = os.path.join(cur_dir, scene)
        normal_dir = os.path.join(scene_dir, "normals")
        masked_colors_dir = os.path.join(scene_dir, "masked_colors")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(masked_colors_dir, exist_ok=True)
        for j in range(num_views):
            view = VIEWS[j]
            normal = normals_pred[j]
            color = images_pred[j]

            normal_filename = f"normals_000_{view}.png"
            rgb_filename = f"rgb_000_{view}.png"
            normal = save_image_to_disk(normal, os.path.join(normal_dir, normal_filename))
            color = save_image_to_disk(color, os.path.join(scene_dir, rgb_filename))

            rm_normal = remove(normal)
            rm_color = remove(color)

            save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
            save_image_numpy(rm_color, os.path.join(masked_colors_dir, rgb_filename))

    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]

    out = images_pred + normals_pred
    return out


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool


def run_3d_generation(scene):
    target_directory = 'instant-nsr-pl'

    # Change the current working directory to the target directory
    os.chdir(target_directory)
    # Path to your Python script
    python_script_path = 'launch.py'

    # Define the arguments
    config_file = 'configs/neuralangelo-ortho-wmask.yaml'
    gpu = '0'
    train_root_dir = f'../outputs/{scene}'
    train_scene = 'scene'

    # Construct the full command
    command = [
        'python', python_script_path,
        '--config', config_file,
        '--gpu', gpu,
        '--train', f'dataset.root_dir={train_root_dir}', f'dataset.scene={train_scene}'
    ]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    # Capture the output and error
    stdout, stderr = process.communicate()

    print("Output:")
    print(stdout)
    print("Error:")
    print(stderr)

def run_mvdiffusion(input_image, save_to=None):
    from utils.misc import load_config
    from omegaconf import OmegaConf
    # parse YAML config to OmegaConf
    cfg = load_config("../configs/mvdiffusion-joint-ortho-6views.yaml")
    # print(cfg)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    pipeline = load_wonder3d_pipeline(cfg)
    torch.set_grad_enabled(False)
    pipeline.to(f'cuda:{_GPU_ID}')

    predictor = sam_init()
    #
    # input_image = gr.Image(type='pil', image_mode='RGBA', height=320, label='Input image', tool=None)
    #
    scale_slider = 3
    steps_slider = 50
    seed = 42
    crop_size = 192
    output_processing = ['Write Results']
    print('Preprocessing...')
    processed_image_highres, processed_image = preprocess(predictor ,input_image=input_image
                                                          ,chk_group=['Background Removal'])
    print(np.asarray(processed_image)[:, :, 3])
    print('Generating...')
    view_1, view_2, view_3, view_4, view_5, view_6, normal_1, normal_2, normal_3, normal_4, normal_5, normal_6 = run_pipeline(
        pipeline ,cfg ,processed_image_highres ,scale_slider, steps_slider ,seed ,crop_size ,output_processing ,save_to)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_to", "--save_to", help="Folder to save outputs")
    parser.add_argument("-img", "--img", help="Folder where input is saved")
    args = parser.parse_args()
    input_image = Image.open(args.img)
    # run_mvdiffusion(input_image,args.save_to)
    run_3d_generation(args.save_to)
