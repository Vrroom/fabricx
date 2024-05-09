# Fabricx

## SpongeCake 

SpongeCake (see citation below) is a popular model that has been used a lot for cloth. I have a work-in-progress implementation in `spongecake_bsdf.py`. In the image below, each row is for a different `alpha`, their roughness parameter, taking values `0.1, 0.5, 1.0`. Each column is for a different `optical_depth`, taking values `1.0, 2.0, 5.0`. 

![grid](https://github.com/Vrroom/fabricx/assets/7254326/7b075ceb-9f5a-4a84-83c7-1f955e8ac30f)

For comparison, this is their render (from Fig. 8).

![image](https://github.com/Vrroom/fabricx/assets/7254326/9bdbd29a-89c5-481d-ad44-b98baa45d8aa)

Note that what we call `optical_depth` is really the product `T\rho` from their equations. This is what they are calling just `T` in their figures. As you can see, ours looks nothing like cloth at the moment, whereas, I think theirs does. Ours becomes darker as `alpha` increases for some reason. 

```
@article{wang2022spongecake,
  title={Spongecake: A layered microflake surface appearance model},
  author={Wang, Beibei and Jin, Wenhua and Ha{\v{s}}an, Milo{\v{s}} and Yan, Ling-Qi},
  journal={ACM Transactions on Graphics (TOG)},

  volume={42},
  number={1},
  pages={1--16},
  year={2022},
  publisher={ACM New York, NY}
}
```
