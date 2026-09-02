# S2CLNet

Official implementation of **S2CLNet: Structure-Constrained Semantic Contrastive Learning for Referring Remote Sensing Image Segmentation**.

![S2CLNet pipeline](pipeline.png)

This repository contains two implementations used in the paper:

```text
clip/   CLIP visual-language implementation
swin/   Swin Transformer + BERT implementation
```

The repository contains model code and dataset adapters only. Dataset files, experiment outputs, and large pretrained weights are excluded from Git.

## Environment

The code was developed with Python 3.7, PyTorch 1.7.1, and CUDA 10.2.

```bash
conda create -n s2clnet python=3.7
conda activate s2clnet
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Downloads

### Datasets

S2CLNet is evaluated on three publicly released referring remote sensing image segmentation datasets. Please obtain each dataset from its original project and follow the corresponding license and preparation instructions.

| Dataset | Official project |
| --- | --- |
| RefSegRS | [AI4EO RRSIS project](https://gitlab.lrz.de/ai4eo/reasoning/rrsis) |
| RRSIS-D | [RMSIN repository](https://github.com/Lsan2401/RMSIN) |
| RISBench | [CroBIM repository](https://github.com/HIT-SIRS/CroBIM) |

Place the downloaded data under the shared dataset directory:

```text
datasets/
├── RefSegRS/
├── rrsisd/
└── RISBench/
```

The datasets were released by their original authors and are not distributed by this repository.

### Pretrained Weights

The CLIP RN101 and Swin-B initialization weights are provided together through Baidu Netdisk:

> [Download pretrained weights](https://pan.baidu.com/s/1j05JQ9_eIWLK3sDkIkuu2A)<br>
> Extraction code: `scln`

After downloading, place the files in the following locations:

| Architecture | Weight | Destination |
| --- | --- | --- |
| CLIP | `RN101.pt` | `clip/pretrained_weights/RN101.pt` |
| Swin Transformer | `swin_base_patch4_window12_384_22k.pth` | `swin/pretrained_weights/swin_base_patch4_window12_384_22k.pth` |

The Swin implementation also uses BERT-base-uncased. Tokenizer metadata is included in `swin/bert-base-uncased/`; additional BERT checkpoint files can be obtained from the [official Hugging Face model page](https://huggingface.co/bert-base-uncased).

## CLIP Implementation

```bash
cd clip
python train.py --dataset refsegrs --refer_data_root ../datasets/RefSegRS --pretrained_clip_weights ./pretrained_weights/RN101.pt
python test.py --dataset refsegrs --refer_data_root ../datasets/RefSegRS --pretrained_clip_weights ./pretrained_weights/RN101.pt --resume ./checkpoints/S2CLNet/model_best_S2CLNet.pth
```

The CLIP model, tokenizer, data adapters, loss, and training utilities are contained in `clip/`.

## Swin-BERT Implementation

```bash
cd swin
python train.py --dataset refsegrs --refer_data_root ../datasets/RefSegRS --ck_bert ./bert-base-uncased --bert_tokenizer ./bert-base-uncased --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth
python test.py --dataset refsegrs --refer_data_root ../datasets/RefSegRS --ck_bert ./bert-base-uncased --bert_tokenizer ./bert-base-uncased --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --resume ./checkpoints/S2CLNet-SwinBERT/model_best_S2CLNet-SwinBERT.pth
```

The Swin visual backbone, BERT source implementation, BERT tokenizer metadata, model components, and data adapters are contained in `swin/`.

## Citation

Please cite the S2CLNet paper when using this code. The implementation builds on the open-source LAVT, Swin Transformer, BERT, and OpenAI CLIP projects.

