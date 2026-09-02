# S2CLNet

Official implementation of **S2CLNet: Structure-Constrained Semantic Contrastive Learning for Referring Remote Sensing Image Segmentation**.

![S2CLNet pipeline](pipeline.png)

This repository provides the model, loss functions, data adapters, and training/evaluation entry points. It does not claim ownership of, or introduce, any dataset.

## Environment

The code was developed with Python 3.7, PyTorch 1.7.1, and CUDA 10.2. Create an environment and install the listed dependencies:

```bash
conda create -n s2clnet python=3.7
conda activate s2clnet
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

The model uses an OpenAI CLIP RN101 checkpoint. Download it separately and place it at `pretrained_weights/RN101.pt`, or pass another location with `--pretrained_clip_weights`. Model checkpoints and datasets are intentionally excluded from Git.

## Model variants

The paper evaluates the proposed method with both CLIP and Swin Transformer visual backbones. The main CLIP training and testing implementation is in the repository root. The Swin backbone definition is available at `lib/backbone_swin.py`; its complete Swin+BERT experiment code is kept separately in the `swin/` implementation directory when needed. The repository includes directory placeholders for `refer/data/`, `pretrained_weights/`, and `swin/pretrained_weights/`, while datasets and large pretrained weight files must be downloaded separately.

## Swin Transformer + BERT

The paper also evaluates the model with a Swin Transformer visual backbone and BERT language encoder. The complete implementation is under `swin/`, with BERT source code in `swin/bert/` and tokenizer metadata in `swin/bert-base-uncased/`.

Download these two required pretrained weights separately (large binary files are not stored in Git):

- CLIP RN101: `pretrained_weights/RN101.pt`
- Swin-B: `swin/pretrained_weights/swin_base_patch4_window12_384_22k.pth`

The dataset directory skeleton is included under `refer/data/`; place the original RRSIS-D or RefSegRS dataset files there.

To run the Swin version, execute from `swin/` and pass local BERT and Swin weight paths to `train.py` or `test.py`.

## Data

The code includes adapters for `rrsisd` and `refsegrs`. Download these datasets from their original sources and pass the corresponding root with `--refer_data_root`.

For RRSIS-D, the root should contain:

```text
refer/data/rrsisd/
├── refs(unc).p
├── instances.json
└── images/rrsisd/JPEGImages/
```

For RefSegRS, the root should contain `images/`, `masks/`, and `output_phrase_train.txt`, `output_phrase_val.txt`, and `output_phrase_test.txt`.

## Training

The supplied entry point runs one CUDA process. Example for RRSIS-D:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset rrsisd \
  --refer_data_root ./refer/data \
  --pretrained_clip_weights ./pretrained_weights/RN101.pt \
  --model_id S2CLNet \
  --epochs 60 \
  --img_size 480
```

Checkpoints are written to `--output-dir` and are ignored by Git.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset rrsisd \
  --refer_data_root ./refer/data \
  --pretrained_clip_weights ./pretrained_weights/RN101.pt \
  --resume ./checkpoints/S2CLNet/model_best_S2CLNet.pth \
  --split val \
  --img_size 480
```

## Citation and Acknowledgements

Please cite the S2CLNet paper when using this code. The implementation builds on the open-source LAVT project and the OpenAI CLIP model.
