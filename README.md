# PCR-CG[[Paper]]( https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700439.pdf)[[Video]](https://youtu.be/gYbgxEV9RHI)

This repository represents the official implementation of the  paper:PCR-CG: Point Cloud Registration via Color and Geometry.

Yu Zhang, Junle Yu, Xiaolin Huang,  Wenhui Zhou, [Ji Hou](https://sekunde.github.io/)

In ECCV-2022 

<img src="/assets/teaser.png" alt="teaser.png" width="90%" />

## Introduction 

### Install
The code is build upon [predator](https://github.com/prs-eth/OverlapPredator).
Follow the predator to install the corresponding Python library:


### Data:3DLoMatch and 3DMatch(Indoor)
#### RGBD DATA
Check the offical [3DMatch](https://3dmatch.cs.princeton.edu/) webiste to download the full rgb-d data.Our preprocess data will be uploaded in few days.
Follow the [Pri3d](https://github.com/Sekunde/Pri3D) to download the well pretrained 2d backbone. Or use the the image-net pretrained model direcctly,you can get a 
near result.
### Generating explicit 2d correspondences and Dump  image matches :
[Superglue](https://github.com/magicleap/SuperGluePretrainedNetwork) offical repository

In our paper,we use two imgs and we choose the first and last image of 50 images(Each 3DMatch point cloud is fused by 50 frames of images).

Generate the 2d matches npz for train/val/test dataset.

Prepocee all the image pairs in text file and dump the keypoints and matches to compressed numpy `npz` files. 
```sh
./match_pairs.py
```

Our preprocessed data will be uploaded in few days.

### Pretrain
to be released

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

## [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) backbone: code will be released in few days
We evaluate our method on  GeoTransformer backbone on the standard 3DMatch/3DLoMatch benchmarks.

|      Methods          | Benchmark   |  Registration Recall   |
| :--------       | :---:       |    :---: | 
|GeoTransformer-lgr| 3DMatch     | 91.5  |
|GeoTransformer-lgr| 3DLoMatch   | 74.0 |
|GeoTransformer-ours-lgr| 3DMatch     | 92.5  |
|GeoTransformer-ours-lgr| 3DLoMatch   | 76.4 |

### Citation
If you find our work helpful, please consider citing
```
@article{zhangpcr,
  title={PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry},
  author={Zhang, Yu and Yu, Junle and Huang, Xiaolin and Zhou, Wenhui and Hou, Ji}
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

