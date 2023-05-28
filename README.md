# PCR-CG: PCR-CG: Point Cloud Registration via Color and Geometry. [ECCV 2022]

Yu Zhang, Junle Yu, Xiaolin Huang,  Wenhui Zhou, [Ji Hou](https://sekunde.github.io/)

<img src="/assets/teaser.png" alt="teaser.png" width="90%" />

[[Paper]]( https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700439.pdf)  [[Video]](https://youtu.be/gYbgxEV9RHI)

This repository represents the official implementation

## Introduction 

### Install
The code is build upon [predator](https://github.com/prs-eth/OverlapPredator).
Follow the predator to install the corresponding Python library:


### Data:3DLoMatch and 3DMatch(Indoor)
#### Download our preprocessed data
 [BaiduNetDisk](https://pan.baidu.com/s/1y65tppzidAH-zvNgzS-A_A)
 
Password：axvz

RGB-D DATA: unzip all the 3dmatch-image.rar files into one directory and change the target image path. The full RGB-D DATA occupies 130+ GB, thus we pick 5 frames from 50 frames evenly and upload this file.

SuperGlue preprocessed data: unzip the dump-match-pairs-160.zip file.

#### RGBD DATA
Check the offical [3DMatch](https://3dmatch.cs.princeton.edu/) webiste to download the full rgb-d data.Our preprocess data will be uploaded in few days.
Follow the [Pri3d](https://github.com/Sekunde/Pri3D) to download the well pretrained 2d backbone. Or use the the image-net pretrained model direcctly,you can get a 
near result.
### Generating explicit 2d correspondences and Dump  image matches :
[Superglue](https://github.com/magicleap/SuperGluePretrainedNetwork) offical repository

In our paper,we use two imgs and we choose the first and last image of 50 images(Each 3DMatch point cloud is fused by 50 frames of images).

Generate the 2d matches npz for train/val/test dataset.

Prepocee all the image pairs in text file and dump the keypoints and matches to compressed numpy `npz` files.  Follow Superglue to dump the images pairs using 'match_pairs.py'
```
python  match_pairs.py
```



### Train
After creating the virtual environment and downloading the corresponding data, PCR-CG can be trained using:
```shell
python main.py configs/train/indoor.yaml
```
modify the img_path and dataset path in configs/train/indoor.yaml as yours.


### Evaluate
```shell
python main.py configs/test/indoor.yaml
```
Then using RANSAC to get the estimated transformation:
modify the source_path and res_path
```shell
sh run_ransac.sh
```



### Citation
If you find our work helpful, please consider citing
```
@inproceedings{zhang2022pcr,
  title={PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry},
  author={Zhang, Yu and Yu, Junle and Huang, Xiaolin and Zhou, Wenhui and Hou, Ji},
  booktitle={European Conference on Computer Vision},
  pages={443--459},
  year={2022},
  organization={Springer}
}
```

---
### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 
- [predator](https://github.com/prs-eth/OverlapPredator) (3D backbone)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences) (3D backbone)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) (3D backbone)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) (Extract 2d correspondences)
- [Pri3d](https://github.com/Sekunde/Pri3D) (Projection Module and 2D backbone)


 We thank the respective authors for open sourcing their methods.

