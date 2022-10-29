# DIP

Official PyTorch implementation of paper:

[**DIP: Deep Inverse Patchmatch for High-Resolution Optical Flow**](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_DIP_Deep_Inverse_Patchmatch_for_High-Resolution_Optical_Flow_CVPR_2022_paper.pdf), **CVPR 2022**


## Installation

Our code is based on pytorch 1.6.0, CUDA 10.1 and python 3.8.


## Demos

All pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1EVrsgk4i6Q_8pxgI2oqgtr8Fe_JWLyQn?usp=sharing).



You can run a trained model on a sequence of images and visualize the results:

```
CUDA_VISIBLE_DEVICES=0 python demo.py \
--model DIP_sintel.pth
```

## Datasets

The datasets used to train and evaluate DIP are as follows:

* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) 
