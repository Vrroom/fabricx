# Fabricx

This is a working collection of tools to model the appearance of fabrics. Most of the code is based on Mitsuba and we occasionally use Blender to visualize things.

## Gallery 

![fine_cloth](https://github.com/Vrroom/fabricx/assets/7254326/b84afd55-b026-4f83-b63d-0b53ed1e2567)
![table_cloth](https://github.com/Vrroom/fabricx/assets/7254326/61d5e9ff-b20c-4149-8531-c428b5d7e4c9)
![rose_cloth](https://github.com/Vrroom/fabricx/assets/7254326/48126913-3790-4944-8793-ac7dc5ac5f21)
![blue_red](https://github.com/Vrroom/fabricx/assets/7254326/601bdd56-9d7c-41da-943d-b93498333829)
![red_black](https://github.com/Vrroom/fabricx/assets/7254326/c5c03ad7-b253-4f68-a526-d052ab96b8be)


## SpongeCake 

SpongeCake (see citation below) is a popular BSDF that has been used a lot for cloth. I have a work-in-progress implementation in `spongecake_bsdf.py`. 

### Surface SGGX Distribution

In the image below, each row is for a different `alpha`, their roughness parameter, taking values `0.1, 0.5, 1.0`. Each column is for a different `optical_depth`, taking values `1.0, 3.0, 5.0`. Please note that we use the surface version of the SGGX distribution function i.e. `S = diag([alpha**2, alpha**2, 1])` here.

![image](https://github.com/Vrroom/fabricx/assets/7254326/041dd93f-66ab-4fe1-863c-e3717fb8189e)

(Why is the shading gone in the last row!!!)

For comparison, this is their render (from Fig. 8).

![image](https://github.com/Vrroom/fabricx/assets/7254326/9bdbd29a-89c5-481d-ad44-b98baa45d8aa)

Note that what we call `optical_depth` is really the product `T\rho` from their equations. This is what they are calling just `T` in their figures.

### Fiber SGGX Distribution

We redid the same image using the fiber version of the SGGX distribution function i.e. `S = diag([1, 1, alpha**2])`. 

![image](https://github.com/Vrroom/fabricx/assets/7254326/7239e6a2-a736-4c6d-bdff-126d6adf7559)

For comparison, the following figure is from the original paper:

![image](https://github.com/Vrroom/fabricx/assets/7254326/799bc42e-c9c3-45e5-b8a3-4c3248629084)

Note that qualitatively, we get some fuzz that can be seen in their renders. This is promising. 

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

## More Images For Fun

![grid_surface](https://github.com/Vrroom/fabricx/assets/7254326/d98ada88-1b13-44d0-b9c2-3cb448bc7027)

![grid_fiber](https://github.com/Vrroom/fabricx/assets/7254326/ea27b3ba-2c58-45c8-8a8d-6fcbc33b79b4)

![grid_surface](https://github.com/Vrroom/fabricx/assets/7254326/19b4e532-ef05-4358-829b-e55ddba7cda0)

![grid_fiber](https://github.com/Vrroom/fabricx/assets/7254326/a51e1eb7-c4cd-46b9-af20-28ff77dd9ebf)

![grid_surface](https://github.com/Vrroom/fabricx/assets/7254326/b0e6e157-342c-43f5-8e44-923be409e0fa)

![grid_fiber](https://github.com/Vrroom/fabricx/assets/7254326/03de6b63-20a1-45aa-85a1-c6e1ded29c45)

(UV map seems wrong)

(Why does this become black at low roughness and high thickness)

Teapots: 

![grid_surface](https://github.com/Vrroom/fabricx/assets/7254326/39e3aee9-a188-4404-9c98-80c136b9e16f)

![grid_fiber](https://github.com/Vrroom/fabricx/assets/7254326/564c75af-9720-4643-91e3-a5b8cc4bcd05)

Actual Cloth Like Models: 

![grid_surface](https://github.com/Vrroom/fabricx/assets/7254326/91ffc55f-b6e7-47d6-9b13-c1cb551f93a3)

![grid_fiber](https://github.com/Vrroom/fabricx/assets/7254326/a3511cf9-e71f-4f80-b372-139c2425ffb8)


## Some debugging 

I was expecting that at `alpha = 1`, sampling from SGGX will be the same as sampling from a uniform sphere. This is not what is happening in practice. Compare the two rows below (first one is with uniform sphere sampling and the other is SGGX with `alpha = 1`). They don't look the same. Funnily, neither of them look like the original paper either. 

(Ok, they look the same when I leave D as it is and not divide it by 4). 

![grid_surface](https://github.com/Vrroom/fabricx/assets/7254326/9d7590cb-006b-4fc9-86e8-7f37ce0a2147)

![grid_surface_2](https://github.com/Vrroom/fabricx/assets/7254326/db62de41-4f15-4594-8cf0-c8e8b8f585c0)

The following is when I used the wrong frame for sampling. These experiments show that figuring out a principled way to do SGGX sampling will solve my problem. 

![grid_surface_2](https://github.com/Vrroom/fabricx/assets/7254326/ea74c576-5d9d-4fc9-a49a-f0c2d56f9eaa)

The fact that shading is missing in my renders is a crucial point. I think this is a good handle for debugging since I can think about it reasonably.

I may be missing the diffuse ??? Not sure. There is some stuff on this in the SGGX paper as well as https://github.com/taqu/Microflake-SGGX/blob/master/microflake_sggx.cpp. Just don't understand the theory well enough at the moment. 


## Questions

* How do you actually sample the `attenuation` for the multi-layered, single-scattering SpongeCake model? 
* Is the orientation map the same as a tangent map? 
