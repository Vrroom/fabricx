# Fabricx

This is a working collection of tools to model the appearance of fabrics. Most of the code is based on Mitsuba and we occasionally use Blender to visualize things.

## Meso-Scale BSDF

We implemented a preliminary version of the Meso-Scale BSDF, as introduced in Zhu et al.'s (2023) paper, which incorporates visibility (see next section).

Horizontal / Vertical:
| Red / Blue | Blue / Red |
|------------|------------|
![256tiles_rb_mm8_alpha000_v015](https://github.com/user-attachments/assets/24065bde-4e78-4878-aaa8-2c8e37bb4ef6) | ![256tiles_br_mm8_alpha000_v015](https://github.com/user-attachments/assets/9c28c5f7-8b46-400e-b7e0-91305e7a21c5)

Visibility Comparisons (Threshold Value):
|  0.00  |  0.15  |  0.30  |
|--------|--------|--------|
![meso_000](https://github.com/user-attachments/assets/9cd026dc-2830-4ed9-bfc9-faec6e61b049) | ![meso_015](https://github.com/user-attachments/assets/a38060ab-ba2f-4c60-ae0d-e74992fc1ea9) | ![meso_030](https://github.com/user-attachments/assets/9cf6905c-fe40-44b2-9626-b0609fa1082a)

Delta Transmission Comparisons:
|  0.025  |  0.050  |
|---------|---------|
![delta0025_128tiles_mm16_v015](https://github.com/user-attachments/assets/fb303b3b-67ea-4a5c-9159-084548655b45) | ![delta0050_128tiles_mm16_v015](https://github.com/user-attachments/assets/089e68d4-bae3-42f3-8a84-318b733ca815)

Additional Demo:
|         beige         |      light blue       |  magenta / dark blue  |      red / green      |
|-----------------------|-----------------------|-----------------------|-----------------------|
![beige_512tiles_d0025_a03](https://github.com/user-attachments/assets/b41188fb-b5cc-48d8-a93f-00dd01fbb8f8) | ![lightblue_128tiles_d0075_a03](https://github.com/user-attachments/assets/4e0c6b45-d2e3-4af6-9af2-d5f1f08a1c75) | ![magenta_darkblue_256tiles_a01](https://github.com/user-attachments/assets/4b972a56-233b-43df-9dfe-c0425e0c73e0) | ![red_green_512tiles_a00](https://github.com/user-attachments/assets/d61af825-7c4a-48db-adf1-43a39ad9fc5d)

Cylinder with Meso-Scale BSDF:
![cylinder_mm256](https://github.com/user-attachments/assets/2171a6e4-d0f1-4a8b-83ab-b27ad6dfaa06)

## Visibility

We fit ASGs to the per point visibility function and discarded light paths that reached an invisible region. This experiment is a bit hacky and needs to be improved. 

![image](https://github.com/Vrroom/fabricx/assets/7254326/689419a3-53d7-4107-90bc-3fa040e9529b)

```
@article{wu2011physically,
  title={Physically-based interactive bi-scale material design},
  author={Wu, Hongzhi and Dorsey, Julie and Rushmeier, Holly},
  journal={ACM Transactions on Graphics (TOG)},
  volume={30},
  number={6},
  pages={1--10},
  year={2011},
  publisher={ACM New York, NY, USA}
}
```
```
@article{jimenez2016practical,
  title={Practical real-time strategies for accurate indirect occlusion},
  author={Jim{\'e}nez, Jorge and Wu, Xianchun and Pesce, Angelo and Jarabo, Adrian},
  journal={SIGGRAPH 2016 Courses: Physically Based Shading in Theory and Practice},
  year={2016}
}
```
```
@inproceedings{zhu2023realistic,
  title={A Realistic Surface-based Cloth Rendering Model},
  author={Zhu, Junqiu and Jarabo, Adrian and Aliaga, Carlos and Yan, Ling-Qi and Chiang, Matt Jen-Yuan},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--9},
  year={2023}
}
```


## Gallery 

**Update**: Added Cylinder model and comparison with actual

One of the microstructures is like 300 micron across (the black one, measured perpendicular to the highlight). This is not very accurate because if we rotate warp and weft, then we don't see the same effect.

| Real | Microscope | Render |
|------|------------|--------|
| ![img_thread](https://github.com/Vrroom/fabricx/assets/7254326/15d21c14-959f-4b8c-b833-d2001d6d3367) |![microscope](https://github.com/Vrroom/fabricx/assets/7254326/d849dc86-77cf-4bb4-a125-337f61fc3e43) |![cylinder](https://github.com/Vrroom/fabricx/assets/7254326/dd9f48de-8926-4899-8efc-c0222630aad5) |

**Update**: Added the BRDF from _Woven Fabric Capture From a Single Photo_ by Jin et al. 

![interesting_red](https://github.com/Vrroom/fabricx/assets/7254326/5e2ec724-9073-420d-9aec-9af385957488)
![shimmer_gold](https://github.com/Vrroom/fabricx/assets/7254326/0b37ab15-6947-4491-a5e7-2bde4bc358af)

**Update**: We can do Delta Transmission also now!!

![gold_delta_transmission](https://github.com/Vrroom/fabricx/assets/7254326/33b3e929-717d-464b-898a-6352b7b5bacd)
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

## ASG

We also have an implementation of the Anisotropic Spherical Gaussians which we use to fit the visibility function. Compare our implementation with the original paper below. LGTM!

| A | B | C |
|---|---|---|
|![asg_2](https://github.com/Vrroom/fabricx/assets/7254326/3d91c5ac-229d-422a-b34e-21dc055abdd0)|![asg_3](https://github.com/Vrroom/fabricx/assets/7254326/4efa1ecc-69d6-434f-9b66-23b259b8bc49)|![asg_4](https://github.com/Vrroom/fabricx/assets/7254326/e5f56111-ed18-4b22-99a0-878f9a4699bf)|

![image](https://github.com/Vrroom/fabricx/assets/7254326/aebc3483-0e6b-4a3a-aef7-2702e25b2377)

```
@article{xu2013anisotropic,
  title={Anisotropic spherical gaussians},
  author={Xu, Kun and Sun, Wei-Lun and Dong, Zhao and Zhao, Dan-Yong and Wu, Run-Dong and Hu, Shi-Min},
  journal={ACM Transactions on Graphics (TOG)},
  volume={32},
  number={6},
  pages={1--11},
  year={2013},
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
