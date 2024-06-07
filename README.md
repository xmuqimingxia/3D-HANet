# 3D-HANet: A Flexible 3D Heatmap Auxiliary Network for Object Detection(TGRS) [Paper link](https://ieeexplore.ieee.org/abstract/document/10056279)

3D HANet is a flexible plug-and-play Auxiliary Network. Although our open source code is based on the CasA detector, it can be verified with simple modifications to other detectors.

Please refer to [CasA](https://github.com/hailanyi/CasA) and [OpenPCDET](https://github.com/open-mmlab/OpenPCDet) for the configuration of the code running environment and training of the model.

Replace the [CasA/pcdet](https://github.com/hailanyi/CasA/tree/master/pcdet) folder in CasA with the [3DHANet/pcdet](https://github.com/xmuqimingxia/3D-HANet/tree/main/pcdet) folder provided by us to run the 3DHANet code.



The module of '3-D Heatmap Generator' is in the [3D-HANet/pcdet/models/backbones_2d/map_to_bev/height_compression.py](https://github.com/xmuqimingxia/3D-HANet/blob/main/pcdet/models/backbones_2d/map_to_bev/height_compression.py); GT of 3D heatmap value is in the [3D-HANet/pcdet/models/dense_heads/anchor_head_single.py](https://github.com/xmuqimingxia/3D-HANet/blob/main/pcdet/models/dense_heads/anchor_head_single.py)

## Getting Started
```
conda create -n spconv2 python=3.9
conda activate spconv2
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-5-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython prefetch-generator
```
### Environment we tested

Our released implementation is tested on.
+ Ubuntu 18.04
+ Python 3.6.9 
+ PyTorch 1.8.1
+ Numba 0.53.1
+ [Spconv 1.2.1](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)
+ NVIDIA CUDA 11.1
+ 8x Tesla V100 GPUs

We also tested on.
+ Ubuntu 18.04
+ Python 3.9.13 
+ PyTorch 1.8.1
+ Numba 0.53.1
+ [Spconv 2.1.22](https://github.com/traveller59/spconv) # pip install spconv-cu111
+ NVIDIA CUDA 11.1 
+ 2x 3090 GPUs

### Prepare Dataset 

#### KITTI Dataset

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
CasA
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

Run following command to creat dataset infos:
```
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```



#### Waymo Dataset

```
CasA
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_train_val_test
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_waymo_track_dbinfos_train_cp.pkl
│   │   │── waymo_infos_test.pkl
│   │   │── waymo_infos_train.pkl
│   │   │── waymo_infos_val.pkl
├── pcdet
├── tools
```

Run following command to creat dataset infos:
```
python3 -m pcdet.datasets.waymo.waymo_tracking_dataset --cfg_file tools/cfgs/dataset_configs/waymo_tracking_dataset.yaml 
```

#### Installation

```
git clone https://github.com/xmuqimingxia/3D-HANet.git
cd 3D-HANet
python3 setup.py develop
```

#### Training

```
cd tools
python3 train.py --cfg_file ${CONFIG_FILE}
```

For example, if you train the CasA-V model:

```
cd tools
python3 train.py --cfg_file cfgs/kitti_models/CasA-V.yaml
```

Multiple GPU train: you can modify the gpu number in the dist_train.sh and run
```
sh dist_train.sh
```
The log infos are saved into log.txt
You can run ```cat log.txt``` to view the training process.

## Acknowledgement
This repo is developed from `OpenPCDet 0.3`, we thank shaoshuai shi for his implementation of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and Hai wu for [CasA](https://github.com/hailanyi/CasA).
