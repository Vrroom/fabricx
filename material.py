from matplotlib import pyplot as plt
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
from mitsuba import ScalarTransform4f as T
from bsdf import * 

mi.register_bsdf('tinted_dielectric_bsdf', lambda props: TintedDielectricBSDF(props))
mi.register_bsdf('disney_diffuse_bsdf', lambda props: DisneyDiffuseBSDF(props))
mi.register_bsdf('disney_metal_bsdf', lambda props: DisneyMetalBSDF(props))

# my_bsdf = mi.load_dict({
#     'type' : 'tinted_dielectric_bsdf',
#     'tint' : [0.2, 0.9, 0.2],
#     'eta' : 1.33
# })

SPP = 1024
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
params = mi.traverse(scene)

image = mi.render(scene, sensor=load_sensor()) 

plt.imshow(image ** (1.0 / 2.2))
plt.savefig('render.png')

