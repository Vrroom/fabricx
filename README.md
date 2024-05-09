# Fabricx

## SpongeCake 

SpongeCake (see citation below) is a popular BSDF that has been used a lot for cloth. I have a work-in-progress implementation in `spongecake_bsdf.py`. 

### Surface SGGX Distribution

In the image below, each row is for a different `alpha`, their roughness parameter, taking values `0.1, 0.5, 1.0`. Each column is for a different `optical_depth`, taking values `1.0, 2.0, 5.0`. Please note that we use the surface version of the SGGX distribution function i.e. `S = diag([alpha**2, alpha**2, 1])` here.

![grid](https://github.com/Vrroom/fabricx/assets/7254326/7b075ceb-9f5a-4a84-83c7-1f955e8ac30f)

For comparison, this is their render (from Fig. 8).

![image](https://github.com/Vrroom/fabricx/assets/7254326/9bdbd29a-89c5-481d-ad44-b98baa45d8aa)

Note that what we call `optical_depth` is really the product `T\rho` from their equations. This is what they are calling just `T` in their figures. As you can see, ours looks nothing like cloth at the moment, whereas, I think theirs does. Ours becomes darker as `alpha` increases for some reason. Also, I'm not sure what `optical_depth` does for us as across each row, the renders look roughly the same.

### Fiber SGGX Distribution

We redid the same image using the fiber version of the SGGX distribution function i.e. `S = diag([1, 1, alpha**2])`. 

![grid](https://github.com/Vrroom/fabricx/assets/7254326/b47ecb47-dd22-4e46-96eb-cc7dda1d18d2)

For comparison, the following figure is from the original paper:

![image](https://github.com/Vrroom/fabricx/assets/7254326/799bc42e-c9c3-45e5-b8a3-4c3248629084)

Note that qualitatively, we get some fuzz that can be seen in their renders. This is promising, if only slightly!  

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
