# 3D-HANet: A Flexible 3D Heatmap Auxiliary Network for Object Detection [Paper link](https://ieeexplore.ieee.org/abstract/document/10056279)

3D HANet is a flexible plug-and-play Auxiliary Network. Although our open source code is based on the CasA detector, it can be verified with simple modifications to other detectors.

Please refer to [CasA](https://github.com/hailanyi/CasA) and [OpenPCDET](https://github.com/open-mmlab/OpenPCDet) for the configuration of the code running environment and training of the model.

Replace the [CasA/pcdet](https://github.com/hailanyi/CasA/tree/master/pcdet) folder in CasA with the [3DHANet/pcdet](https://github.com/xmuqimingxia/3D-HANet/tree/main/pcdet) folder provided by us to run the 3DHANet code.



The module of '3-D Heatmap Generator' is in the [3D-HANet/pcdet/models/backbones_2d/map_to_bev/height_compression.py](https://github.com/xmuqimingxia/3D-HANet/blob/main/pcdet/models/dense_heads/anchor_head_single.py); GT of 3D heatmap value is in the [3D-HANet/pcdet/models/dense_heads/anchor_head_single.py](https://github.com/xmuqimingxia/3D-HANet/pcdet/models/dense_heads/anchor_head_single.py)
