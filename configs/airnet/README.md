# AirNet (CVPR'2022)

> [All-In-One Image Restoration for Unknown Corruption (AirNet)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf)

> **Task**: Image Restoration

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In this paper, we study a challenging problem in image
restoration, namely, how to develop an all-in-one method
that could recover images from a variety of unknown corruption types and levels. To this end, we propose an All-inone Image Restoration Network (AirNet) consisting of two
neural modules, named Contrastive-Based Degraded Encoder (CBDE) and Degradation-Guided Restoration Network (DGRN). The major advantages of AirNet are twofold. First, it is an all-in-one solution which could recover
various degraded images in one network. Second, AirNet is
free from the prior of the corruption types and levels, which
just uses the observed corrupted image to perform inference. These two advantages enable AirNet to enjoy better flexibility and higher economy in real world scenarios
wherein the priors on the corruptions are hard to know and
the degradation will change with space and time. Extensive experimental results show the proposed method outperforms 17 image restoration baselines on four challenging
datasets.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/43229734/204189325-f4dae3f1-0cc8-4545-aa40-fe908f1e73b2.png" width="400"/>
</div >

## Results and models

|                                                       Method                                                        |    Model    |      PSNR      |      SSIM      | GPU Info |            Download             |
| :-----------------------------------------------------------------------------------------------------------------: | :---------: | :------------: | :------------: | :------: | :-----------------------------: |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py) | All-in-one  | 34.8992(34.90) | 0.9689(0.9675) | 1 (A100) | [model](<>) \| log(coming soon) |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py) | Derain only | 34.8963(34.90) | 0.9676(0.9657) | 1 (A100) | [model](<>) \| log(coming soon) |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots.py) | All-in-one  | 27.8220(27.94) | 0.9616(0.9615) | 1 (A100) | [model](<>) \| log(coming soon) |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots.py) | Dehaze only | 23.1438(23.18) | 0.9023(0.9000) | 1 (A100) | [model](<>) \| log(coming soon) |

Note:

- a(b) where a denotes the value run by MMEditing, b denotes the value copied from the original paper.
- PSNR is evaluated on RGB channels.
- SSIM is evaluated by averaging SSIMs on RGB channels.

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py

# single-gpu train
python tools/train.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py

# multi-gpu train
./tools/dist_train.sh configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py /path/to/checkpoint

# single-gpu test
python tools/test.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py /path/to/checkpoint

# multi-gpu test
./tools/dist_test.sh configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py /path/to/checkpoint 8
```

Pretrained checkpoints will come soon.

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@inproceedings{AirNet,
author = {Li, Boyun and Liu, Xiao and Hu, Peng and Wu, Zhongqin and Lv, Jiancheng and Peng, Xi},
title = {{All-In-One Image Restoration for Unknown Corruption}},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
year = {2022},
address = {New Orleans, LA},
month = jun
}
```
