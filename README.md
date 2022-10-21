# PCR-CG

This repository represents the official implementation of the paper:PCR-CG: Point Cloud Registration via Color and Geometry

## Introduction 
The code is build upon [predator](https://github.com/prs-eth/OverlapPredator).
Follow the predator to install the corresponding Python library:


## Data
Check the offical [3DMatch](https://3dmatch.cs.princeton.edu/) webiste to download the full rgb-d data.
Follow the [Pri3d](https://github.com/Sekunde/Pri3D) to download the well pretrained 2d backbone. Or use the the image-net pretrained model direcctly,you can get a 
near result.
## utlize Superglue for generating explicit 2d correspondences:
[Superglue](https://github.com/magicleap/SuperGluePretrainedNetwork) offical repository

In our paper,we use two imgs and we choose the first and last image of 50 images(Each 3DMatch point cloud is fused by 50 frames of images).

Generate the 2d matches npz for train/val/test dataset.
### Dump 2d Matches 

The simplest usage of this script will process the image pairs listed in a given text file and dump the keypoints and matches to compressed numpy `npz` files. 
```sh
./match_pairs.py
```

The resulting `.npz` files can be read from Python as follows:

```python
>>> import numpy as np
>>> path = 'dump_match_pairs/scene0711_00_frame-001680_scene0711_00_frame-001995_matches.npz'
>>> npz = np.load(path)
>>> npz.files
['keypoints0', 'keypoints1', 'matches', 'match_confidence']
>>> npz['keypoints0'].shape
(382, 2)
>>> npz['keypoints1'].shape
(391, 2)
>>> npz['matches'].shape
(382,)
>>> np.sum(npz['matches']>-1)
115
>>> npz['match_confidence'].shape
(382,)
our preprocessed data will be uploaded in few days.

### 3DLoMatch and 3DMatch(Indoor)
#### Train
After creating the virtual environment and downloading the datasets, Predator can be trained using:
```shell
python main.py configs/train/indoor.yaml
```
modify the img_path and dataset path in configs/train/indoor.yaml

#### Evaluate
```shell
python main.py configs/test/indoor.yaml
```
Then using RANSAC to get the estimated transformation:
modify the source_path and res_path
```shell
sh run_ransac.sh
```

## [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) backbone: Geotransformer code will be updated in few days
We evaluate our method on  GeoTransformer backbone on the standard 3DMatch/3DLoMatch benchmarks.

|      Methods          | Benchmark   |  RR   |
| :--------       | :---:       |    :---: | 
|GeoTransformer-lgr| 3DMatch     | 91.5  |
|GeoTransformer-lgr| 3DLoMatch   | 74.0 |
|GeoTransformer-ours-lgr| 3DMatch     | 92.5  |
|GeoTransformer-ours-lgr| 3DLoMatch   | 76.4 |
