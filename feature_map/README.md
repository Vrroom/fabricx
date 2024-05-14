# Procedural Feature Map Generator

This is a feature map generator that can be used for parameterizing a spatially varying BSDF. 

## Usage 

We'll keep adding more and more options as required. 

```
$ ./main --help
 - Command line options
Usage:
  ./main [OPTION...]

  -w, --window-width arg        Image width (default: 1024)
  -h, --window-height arg       Image height (default: 1024)
  -p, --num-profile-points arg  Number of points on the profile curve 
                                (default: 200)
  -s, --num-sweep-points arg    Number of points on sweep curve (default: 
                                200)
  -r, --radius arg              Radius of curvature (default: 10.000000)
      --phi arg                 Twisting angle in radians (default: 
                                1.047198)
  -f, --file-path arg           File path for input/output
      --help                    Print help
```

## `.fil` file

This file specifies a filament map in the form `x y w h type` where `x`, `y` are the origin of a filament. `w`, `h` are dimensions and `type` is 0 or 1 representing _warp_ or _weft_ thread type. See `twill.fil`, `satin.fil` and `plain.fil` for examples. 

## Examples

Normal Maps, Tangent Maps and ID maps for 3 different weave patterns. 

![img](https://github.com/Vrroom/fabricx/assets/7254326/06ea91e2-89c1-4894-8b32-3c77f76e6041)

Please compare with the following from Jin et al.

![image](https://github.com/Vrroom/fabricx/assets/7254326/2a69877f-b382-429a-bc9f-8fbbf88b4913)

## References

```
@article{irawan2012specular,
  title={Specular reflection from woven cloth},
  author={Irawan, Piti and Marschner, Steve},
  journal={ACM Transactions on Graphics (TOG)},
  volume={31},
  number={1},
  pages={1--20},
  year={2012},
  publisher={ACM New York, NY, USA}
}

@inproceedings{jin2022woven,
  title={Woven fabric capture from a single photo},
  author={Jin, Wenhua and Wang, Beibei and Hasan, Milos and Guo, Yu and Marschner, Steve and Yan, Ling-Qi},
  booktitle={SIGGRAPH Asia 2022 conference papers},
  pages={1--8},
  year={2022}
}
```
