# OmniShaper-AI

### Enhance your 3D modeling using Deep Learning

### Requirements:
- Python 3.9
- CUDA 11.8
### Installation
```bash
git clone '...'
cd OmniShaper-AI
python -m venv venv
source venv/bin/activate
./scripts/install-requirements.sh
```

### Usage
1) Run the command to generate n images
```python
python -c "from ml_pipeline.prompt_to_img import generate_images; generate_images('prompt', n, request_id, 'optional_negative_prompt')"
```
2) Run the command to generate a 3d object from the generated image
```python
python -c "from ml_pipeline.img_to_3d import generate_3d_model; generate_3d_model(request_id, image_id)"
```

### Example
```python
python -c "from ml_pipeline.prompt_to_img import generate_images; generate_images('realistic dinosaur in cowboy hat. White background', 3, 123, 'crop')"
python -c "from ml_pipeline.img_to_3d import generate_3d_model; generate_3d_model(123, 2)"
```