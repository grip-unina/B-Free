# Training Dataset
[![arXiv](https://img.shields.io/badge/-Data-ffab03.svg?style=for-the-badge&logo=files&logoColor=ffffff)](https://www.grip.unina.it/download/prog/B-Free/training_data/)

You can download our training dataset at the above link.

***Note:*** *training a method on this dataset only partially reflects the **inpainted+** version of our augmentation 
strategy, as the dataset does not include blurring, JPEG, and the extra augmentations of **inpainted++** (scaling, 
cut-out, noise addition, jittering), which are performed at training time.*

|        | Real   | Generated |
|--------|--------|-----------|
| Source | COCO   | SD 2.1    |
| Num.   | 51,517 | 309,102   |  

To create this dataset we used images collected from the training set of MS-COCO dataset (discarding images with 
licenses different than Creative Commons) and images generated with Stable Diffusion 2.1.\
Specifically, we first extracted the largest central crop and resize it to 512 x 512.
Then, for the generations, we used the inpainting code from the official [Stable Diffusion 2.1 repository](https://github.com/Stability-AI/stablediffusion).
Note that we did NOT embed the watermark during generation (`put_watermark` function).

The dataset at the above link contains the following folders and files:
- **COCO_real_512**: real images from COCO (largest central crop resized to 512x512)
- **SD2.1_selfconditioned**: self-conditioned images
- **SD2.1_selfconditioned_origBG**: self-conditioned images with original background restored
- **SD2.1_inpainted_samecat**: an object is replaced with one of the same category
- **SD2.1_inpainted_samecat_origBG**: same as *inpainted_samecat*, but with original background restored
- **SD2.1_inpainted_diffcat**: an object is replaced with one of a different category
- **SD2.1_inpainted_diffcat_origBG**: same as *inpainted_diffcat*, but with original background restored
- **mask**: object masks used for the generation of *inpainted_samecat* and for the background replacement of *inpainted_samecat_origBG* and *selfconditioned_origBG*
- **bbox**: bounding boxes used for the generation of *inpainted_diffcat* and for the background replacement of *inpainted_diffcat_origBG*
- **train_list.csv**: contains the ids for our training split and extra info for each image, such as the object category
- **valid_list.csv**: contains the ids for our validation split and extra info for each image, such as the object category

Note that both *train* and *valid* split come from the MS-COCO 2017 **training set**.

Further details can be found in appendix A of the paper.

## md5sum

- 41741e8e81e61455d01455452fcf15ae COCO_real_512.zip: 
- a96855ced782ce09b6139ba68af94dd0 SD2.1_selfconditioned.zip: 
- b2e8530fa6904bc25e6fe6f1ed45149d SD2.1_selfconditioned_origBG.zip: 
- fc3ba27bc97d60bcf18ca57931aebd3b SD2.1_inpainted_samecat.zip: 
- 466fed55a78dacc1f82bd95f6eb7cd5c SD2.1_inpainted_samecat_origBG.zip: 
- 5b00dc28d18809c2efc943c89c3f5514 SD2.1_inpainted_diffcat.zip: 
- d289c6b0fe4b19f229f4578e61ca8032 SD2.1_inpainted_diffcat_origBG.zip: 
- 0294cb1611ad1c50af72d6cace3492ca masks_and_bbox.zip
- 06581da53bf81741501396d71eb7ec5f train_list.csv:
- 72e7b4be2d049029dbedca7e75af91b0 valid_list.csv: 


## Bibtex

If you use this dataset, please cite:

```
@inproceedings{Guillaro2025biasfree,
  title={A Bias-Free Training Paradigm for More General AI-generated Image Detection},
  author={Guillaro, Fabrizio and Zingarini, Giada and Usman, Ben and Sud, Avneesh and Cozzolino, Davide and Verdoliva, Luisa},
  booktitle={IEEE/CVF conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```


## License

Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA'). 

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 