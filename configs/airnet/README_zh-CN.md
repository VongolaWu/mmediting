# AirNet (cvpr'2022)

> **任务**: 图像恢复

<!-- [ALGORITHM] -->

<details>
<summary align="right">AirNet (CVPR'2022)</summary>

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

</details>

|                                                        方法                                                         |    模型     |      PSNR      |      SSIM      | GPU 信息 |             下载             |
| :-----------------------------------------------------------------------------------------------------------------: | :---------: | :------------: | :------------: | :------: | :--------------------------: |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py) | All-in-one  | 34.8992(34.90) | 0.9689(0.9675) | 1 (A100) | [模型](<>) \| 日志(即将到来) |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py) | Derain only | 34.8963(34.90) | 0.9676(0.9657) | 1 (A100) | [模型](<>) \| 日志(即将到来) |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots.py) | All-in-one  | 27.8220(27.94) | 0.9616(0.9615) | 1 (A100) | [模型](<>) \| 日志(即将到来) |
| [airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots](/configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_sots.py) | Dehaze only | 23.1438(23.18) | 0.9023(0.9000) | 1 (A100) | [模型](<>) \| 日志(即将到来) |

Note:

- 评估结果a(b)中，a代表由MMEditing测量，b代表由原论文提供。
- PSNR是在RGB通道评估。
- SSIM是平均的分别在RGB通道评估的SSIM。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py

# 单个GPU上训练
python tools/train.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py

# 多个GPU上训练
./tools/dist_train.sh configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py /path/to/checkpoint

# 单个GPU上测试
python tools/test.py configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py /path/to/checkpoint

# 多个GPU上测试
./tools/dist_test.sh configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1500E_rain.py /path/to/checkpoint 8
```

预训练模型未来将会上传，敬请等待。
更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
