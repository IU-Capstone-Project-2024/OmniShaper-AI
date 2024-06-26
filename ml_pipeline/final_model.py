from test_scripts.pipeline import Pipeline
from MicroDreamer.pipeline import ImgTo3dPipeline

def promt_to_3D(promt) -> str:
    promt_to_img =  Pipeline()
    image, variants = promt_to_img('house')

    filepath = 'images/'

    image.save(filepath + 'img.png')
    for i in range(len(variants)):
        variants[i].save(filepath + 'img_' + str(i) + '.png')

    img_to_3d = ImgTo3dPipeline()
    
    img_to_3d('images/img.png', 512)
    
    