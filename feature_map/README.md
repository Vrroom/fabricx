# Procedural Feature Map Generator

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
