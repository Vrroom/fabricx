from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
# mi.set_variant('scalar_rgb')
from mitsuba import ScalarTransform4f as T
from bsdf import * 
from solid_texture_bsdf import * 
from spongecake_bsdf import * 
from itertools import product

mi.register_bsdf('tinted_dielectric_bsdf', lambda props: TintedDielectricBSDF(props))
mi.register_bsdf('disney_diffuse_bsdf', lambda props: DisneyDiffuseBSDF(props))
mi.register_bsdf('disney_metal_bsdf', lambda props: DisneyMetalBSDF(props))
mi.register_bsdf('disney_glass_bsdf', lambda props: DisneyGlassBSDF(props))
mi.register_bsdf('disney_clearcoat_bsdf', lambda props: DisneyClearcoatBSDF(props))
mi.register_bsdf('disney_sheen_bsdf', lambda props: DisneySheenBSDF(props))
mi.register_bsdf('disney_principled_bsdf', lambda props : DisneyPrincipledBSDF(props))
mi.register_bsdf('solid_texture_bsdf', lambda props : SolidTextureBSDF(props))
mi.register_bsdf('spongecake_bsdf', lambda props: SimpleSpongeCake(props))


SPP = 512
WIDTH = 683
HEIGHT = 512
MAX_DEPTH = -1

def load_sensor () : 
    return mi.load_dict({
        'type': 'perspective', 
        'fov_axis': 'smaller', 
        'focus_distance': 6.0, 
        'fov': 28.8415, 
        'to_world': T.look_at(
            origin=[3.69558, -3.46243, 3.25463], 
            target=[3.04072, -2.85176, 2.80939], 
            up=[-0.317366, 0.312466, 0.895346]
            # origin=[-6.0, 0.0, 0.0], 
            # target=[0, 0, 0], 
            # up=[0, 1, 0]
        ), 
        'sampler': {
            'type': 'independent', 
            'sample_count': SPP, 
        }, 
        'film': {
            'type': 'hdrfilm', 
            'width': WIDTH, 
            'height': HEIGHT,
            'pixel_format': 'rgb', 
            'rfilter': {
                'type': 'gaussian'
            }
        }
    })

scene = mi.load_file('matpreview/scene.xml')

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

def make_image_grid (images, row_major=True, gutter=True):
    """
    Make a large image where the images are stacked.

    images: list/list of list of PIL Images
    row_major: if images is list, whether to lay them down horizontally/vertically
    """
    assert isinstance(images, list) and len(images) > 0, "images is either not a list or an empty list"
    if isinstance(images[0], list) :
        return make_image_grid([make_image_grid(row) for row in images], False)
    else :
        if row_major :
            H = min(a.size[1] for a in images)
            images = [a.resize((int(H * a.size[0] / a.size[1]), H)) for a in images]
            W = sum(a.size[0] for a in images)
            gutter_width = int(0.01 * W) if gutter else 0
            W += (len(images) - 1) * gutter_width
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (cSum, 0))
                cSum += (a.size[0] + gutter_width)
        else :
            W = min(a.size[0] for a in images)
            images = [a.resize((W, int(W * a.size[1] / a.size[0]))) for a in images]
            H = sum(a.size[1] for a in images)
            gutter_width = int(0.01 * W) if gutter else 0
            H += (len(images) - 1) * gutter_width
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (0, cSum))
                cSum += (a.size[1] + gutter_width)
        return img

params = mi.traverse(scene)
alpha = [0.1, 0.5, 1.0]
optical_depth = [1.0, 2.0, 5.0]

images = []

for a in alpha: 
    images.append([])
    for od in optical_depth: 
        params['bsdf-matpreview.alpha'] = mi.Float(a)
        params['bsdf-matpreview.optical_depth'] = mi.Float(od)
        params.update()
        image = mi.render(scene, sensor=load_sensor()) 
        img = (image ** (1.0 / 2.2)).numpy()
        img = np.clip(img, 0, 1)
        images[-1].append(imgArrayToPIL(img))

make_image_grid(images).save('grid.png')

