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

mi.register_bsdf('tinted_dielectric_bsdf', lambda props: TintedDielectricBSDF(props))
mi.register_bsdf('disney_diffuse_bsdf', lambda props: DisneyDiffuseBSDF(props))
mi.register_bsdf('disney_metal_bsdf', lambda props: DisneyMetalBSDF(props))
mi.register_bsdf('disney_glass_bsdf', lambda props: DisneyGlassBSDF(props))
mi.register_bsdf('disney_clearcoat_bsdf', lambda props: DisneyClearcoatBSDF(props))
mi.register_bsdf('disney_sheen_bsdf', lambda props: DisneySheenBSDF(props))
mi.register_bsdf('disney_principled_bsdf', lambda props : DisneyPrincipledBSDF(props))
mi.register_bsdf('solid_texture_bsdf', lambda props : SolidTextureBSDF(props))
mi.register_bsdf('spongecake_bsdf', lambda props: SimpleSpongeCake(props))


SPP = 2048
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

params = mi.traverse(scene)

image = mi.render(scene, sensor=load_sensor()) 
img = (image ** (1.0 / 2.2)).numpy()
img = np.clip(img, 0, 1)
imgArrayToPIL(img).save('img.png')

