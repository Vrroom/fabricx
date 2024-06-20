from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import mitsuba as mi
from itertools import product
import argparse

def print_dict_as_table(d):
    # Determine the maximum width of the keys and values for alignment
    max_key_width = max(len(str(key)) for key in d.keys())
    max_value_width = max(len(str(value)) for value in d.values())

    # Print the table header
    print(f"{'Key':<{max_key_width}} | {'Value':<{max_value_width}}")
    print('-' * (max_key_width + max_value_width + 3))

    # Print the key-value pairs
    for key, value in d.items():
        print(f"{str(key):<{max_key_width}} | {str(value):<{max_value_width}}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Let us play with cloth')
    parser.add_argument('--spp', type=int, default=512, help='Samples per pixel')
    parser.add_argument('--scene', type=str, default='cloth/cloth_scene.xml', help='Path to scene')
    parser.add_argument('--debug', action='store_true', help='Whether running in debug mode or not')
    # bsdf parameters
    parser.add_argument('--bsdf-type', type=str, default='SimpleSpongeCake', help='Class name of the bsdf')
    parser.add_argument('--alpha', type=float, nargs='+', default=[1.0], help='List of integers for alpha')
    parser.add_argument('--optical_depth', type=float, nargs='+', default=[2.0], help='List of integers for alpha') # T * \rho = 2.0, as in Jin et al.'s paper, section 3.2
    parser.add_argument('--fiber', action='store_true', help='Whether to use the fiber mode or the surface mode')
    parser.add_argument('--texture', type=str, default=None, help='Texture file')
    parser.add_argument('--normal_map', type=str, default=None, help='Normal map file')
    parser.add_argument('--tangent_map', type=str, default=None, help='Tangent map file')
    parser.add_argument('--perturb_specular', action='store_true', help='Whether to randomly perturb specular weight')
    parser.add_argument('--delta_transmission', action='store_true', help='Whether to use delta transmission')
    # output path
    parser.add_argument('--save_to', type=str, default='img.png', help='Where to save the result to')

    args = parser.parse_args()
    
    print_dict_as_table(vars(args))

    if args.debug: 
        mi.set_variant('scalar_rgb')
        print('Note: We won\'t be able to change the scene parameters now!!')
    else :
        mi.set_variant('llvm_ad_rgb')

    from mitsuba import ScalarTransform4f as T
    from bsdf import * 
    from solid_texture_bsdf import * 
    from spongecake_bsdf import * 
    from utils import * 

    cls_name = globals()[args.bsdf_type]

    # register the family of BSDFs
    mi.register_bsdf('tinted_dielectric_bsdf', lambda props: TintedDielectricBSDF(props))
    mi.register_bsdf('disney_diffuse_bsdf', lambda props: DisneyDiffuseBSDF(props))
    mi.register_bsdf('disney_metal_bsdf', lambda props: DisneyMetalBSDF(props))
    mi.register_bsdf('disney_glass_bsdf', lambda props: DisneyGlassBSDF(props))
    mi.register_bsdf('disney_clearcoat_bsdf', lambda props: DisneyClearcoatBSDF(props))
    mi.register_bsdf('disney_sheen_bsdf', lambda props: DisneySheenBSDF(props))
    mi.register_bsdf('disney_principled_bsdf', lambda props : DisneyPrincipledBSDF(props))
    mi.register_bsdf('solid_texture_bsdf', lambda props : SolidTextureBSDF(props))
    mi.register_bsdf('spongecake_bsdf', lambda props: cls_name(props, \
        normal_map=args.normal_map, tangent_map=args.tangent_map, texture=args.texture, 
        perturb_specular=args.perturb_specular, delta_transmission=args.delta_transmission))

    scene = mi.load_file(args.scene)

    params = mi.traverse(scene)
    if 'bsdf-matpreview.surface_or_fiber' in params: 
        params['bsdf-matpreview.surface_or_fiber'] = mi.Bool(not args.fiber)
        params.update()

    images = []

    for a in args.alpha: 
        images.append([])
        for od in args.optical_depth: 
            if 'bsdf-matpreview.alpha' in params : 
                params['bsdf-matpreview.alpha'] = mi.Float(a)
            if 'bsdf-matpreview.optical_depth' in params :
                params['bsdf-matpreview.optical_depth'] = mi.Float(od)
            params.update()
            image = mi.render(scene, spp=args.spp)
            img = (image ** (1.0 / 2.2)).numpy()
            img = np.clip(img, 0, 1)
            images[-1].append(imgArrayToPIL(img))

    make_image_grid(images).save(args.save_to)
