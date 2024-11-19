# EVA-Gaussian: 3D Gaussian-Based Real-time Human Novel View Synthesis Under Diverse Camera Settings
Yingdong Hu, Zhening Liu, Jiawei Shao, Zehong Lin, Jun Zhang

Hongkong University of Science and Technology

**Welcome to the official code repository for EVA-Gaussian**! This project implements the methods proposed in the paper "EVA-Gaussian: 3D Gaussian-Based Real-time Human Novel View Synthesis Under Diverse Camera Settings."


# EVA-Gaussian: 3D Gaussian-Based Real-time Human Novel View Synthesis Under Diverse Camera Settings
Yingdong Hu, Zhening Liu, Jiawei Shao, Zehong Lin, Jun Zhang

Hongkong University of Science and Technology

**Welcome to the official code repository for EVA-Gaussian**! This project implements the methods proposed in the paper "EVA-Gaussian: 3D Gaussian-Based Real-time Human Novel View Synthesis Under Diverse Camera Settings."

[Video](https://www.bilibili.com/video/BV1SBmBYEEQF/?spm_id_from=333.999.0.0&vd_source=a8e75db414b53dbeb2e224535e04af88)



Paper page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian

Project page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian

## Hardware Requestment

EVA-Gaussian takes around 25 GB memory usage for training at the batch size of 1, which means you will need GPU with at least 28 GB memory.

## Environment Setup
Pytorch with CUDA verson 11.8 is used in EVA-Gaussian according to the requirements of diff-gaussian-splatting. Users may try other verson of CUDA at their need.

```bash
conda create -n eva_gaussian python=3.10
conda activate eva_gaussian
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./feature-splatting
```

## Prepare the Data

### THuman2.0 dataset
See [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian/blob/main/prepare_data/MAKE_DATA.md)' repository for dataset setup.

### THumansit dataset
See [THumansit](https://github.com/jiajunzhang16/ins-hoi) for dataset downloading. The setup procedure is the same with THuman2.0 dataset.

## Easy Training (without anchor loss)

Firstly pretrain the Gaussian position estimation network with GT depth:

```bash
python depthnet_pretrain.py
```

Then train the whole EVA-Gaussian with the pretrained weight:

```bash
python train.py
```

## Training with anchor loss

### Anchor loss setup

Use process.py for dataset processing and landmark generation. This process require extra package including: mmpose, mmdet, mmcv, xtcocotools and face_recognition.

Place the landmark.json file in your dataset, in the same level directory as the train/val folder.

Set dataset.anchor=True in your config.

### Train

```bash
python depthnet_pretrain.py
```

Then train the whole EVA-Gaussian with the pretrained weight:

```bash
python train.py
```

## Citation

```
@article{hu2024eva,
  title={EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Camera Settings},
  author={Hu, Yingdong and Liu, Zhening and Shao, Jiawei and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2410.01425},
  year={2024}
}
```


Paper page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian

Project page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian

## Hardware Requestment

EVA-Gaussian takes around 25 GB memory usage for training at the batch size of 1, which means you will need GPU with at least 28 GB memory.

## Environment Setup
Pytorch with CUDA verson 11.8 is used in EVA-Gaussian according to the requirements of diff-gaussian-splatting. Users may try other verson of CUDA at their need.

```bash
conda create -n eva_gaussian python=3.10
conda activate eva_gaussian
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./feature-splatting
```

## Prepare the Data

### THuman2.0 dataset
See [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian/blob/main/prepare_data/MAKE_DATA.md)' repository for dataset setup.

### THumansit dataset
See [THumansit](https://github.com/jiajunzhang16/ins-hoi) for dataset downloading. The setup procedure is the same with THuman2.0 dataset.

## Easy Training (without anchor loss)

Firstly pretrain the Gaussian position estimation network with GT depth:

```bash
python depthnet_pretrain.py
```

Then train the whole EVA-Gaussian with the pretrained weight:

```bash
python train.py
```

## Training with anchor loss

### Anchor loss setup

Use process.py for dataset processing and landmark generation. This process require extra package including: mmpose, mmdet, mmcv, xtcocotools and face_recognition.

Place the landmark.json file in your dataset, in the same level directory as the train/val folder.

Set dataset.anchor=True in your config.

### Train

```bash
python depthnet_pretrain.py
```

Then train the whole EVA-Gaussian with the pretrained weight:

```bash
python train.py
```

## Citation

```
@article{hu2024eva,
  title={EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Camera Settings},
  author={Hu, Yingdong and Liu, Zhening and Shao, Jiawei and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2410.01425},
  year={2024}
}
```