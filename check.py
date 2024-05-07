import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant('scalar_rgb')
from mitsuba import ScalarTransform4f as T

sensor = mi.load_dict({
    'type': 'perspective',
    'fov': 41,
    'fov_axis': 'larger',
    'to_world': T.look_at(
        origin=[2.5, 2.5, 10],  # Camera position
        target=[2.5, 2.5, 0],   # Point to look at
        up=[0, 1, 0]            # Up vector
    ),
    'sampler': {
        'type': 'multijitter',
        'sample_count': 128
    },
    'film': {
        'type': 'hdrfilm',
        'width': 612,
        'height': 512,
        'pixel_format': 'rgb',
        'rfilter': {
            'type': 'gaussian'
        }
    }
        
})
integrator = mi.load_dict({
    'type': 'aov',
    'aovs': 'd:depth,nn:geo_normal,pos:position'
},)

bear = mi.load_dict({
            'type': 'ply',
            'filename': 'pink.ply',
                
            # 'to_world': T.translate([1.7,-1,0]).scale(0.02).rotate(axis=[1,0,0], angle=-90).rotate(axis=[0,0,1], angle=90),
            'flip_normals': True,
            'bsdf': {
                'type': 'diffuse'
            },
})
scene = mi.load_dict({
            'type':'scene',
            'integrator': integrator,
            'sensor': sensor,
            'object': bear,
            
            'light': {'type': 'constant'},   
    
})

from PIL import Image
import numpy as np
import base64, io

ENCODING = 'utf-8'

def make_image_grid (images, row_major=True):
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
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (cSum, 0))
                cSum += a.size[0]
        else :
            W = min(a.size[0] for a in images)
            images = [a.resize((W, int(W * a.size[1] / a.size[0]))) for a in images]
            H = sum(a.size[1] for a in images)
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (0, cSum))
                cSum += a.size[1]
        return img

def imsave (arr, fname) :
    """ utility for saving numpy array """
    imgToPIL(arr).save(fname)

def normed (arr) : 
    return (arr - arr.min()) / (arr.max() - arr.min())

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, np.float64, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [np.int32, np.int64, int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

image = mi.render(scene)

bmp = scene.sensors()[0].film().bitmap().split()[2][1]
bmp_np = np.array(bmp)
imgArrayToPIL(bmp_np / 2. + 0.5).save('normal_map.png')

bmp = scene.sensors()[0].film().bitmap().split()[3][1]
bmp_np = np.array(bmp)
Image.fromarray((normed(bmp_np[..., 2]) * 255.).astype(np.uint8), mode='L').save('bump.png')
