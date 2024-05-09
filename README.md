# Fabricx

This is a working collection of tools to model the appearance of fabrics. Most of the code is based on Mitsuba and we occasionally use Blender to visualize things.

## SpongeCake 

SpongeCake (see citation below) is a popular BSDF that has been used a lot for cloth. I have a work-in-progress implementation in `spongecake_bsdf.py`. 

### Surface SGGX Distribution

In the image below, each row is for a different `alpha`, their roughness parameter, taking values `0.1, 0.5, 1.0`. Each column is for a different `optical_depth`, taking values `1.0, 3.0, 5.0`. Please note that we use the surface version of the SGGX distribution function i.e. `S = diag([alpha**2, alpha**2, 1])` here.

![image](https://github.com/Vrroom/fabricx/assets/7254326/041dd93f-66ab-4fe1-863c-e3717fb8189e)

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
## SGGX 

Here, we show our implementation of the SGGX distribution function and contrast it with the one in the original paper (see citation below). 

| Surface | Fiber |
|---------|-------|
|![surface](https://github.com/Vrroom/fabricx/assets/7254326/55e72bd2-8182-4b82-88c3-08aac8337a9a) | ![fiber](https://github.com/Vrroom/fabricx/assets/7254326/28bd250b-1e3f-4f39-b632-d92a505b7d6f) |



Original:

![image](https://github.com/Vrroom/fabricx/assets/7254326/48f2f9ff-0db8-4f07-9260-71dc1d0d0cf0)

Note that we get the same bands around the equator for `fiber` and peaks at the poles for `surface`. This is a sanity check that we are on the right track.  

```
@article{heitz2015sggx,
  title={The SGGX microflake distribution},
  author={Heitz, Eric and Dupuy, Jonathan and Crassin, Cyril and Dachsbacher, Carsten},
  journal={ACM Transactions on Graphics (TOG)},
  volume={34},
  number={4},
  pages={1--11},
  year={2015},
  publisher={ACM New York, NY, USA}
}
```
