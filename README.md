# MAP-SAM on ISPRS Vaihingen

This repository contains the MAP-SAM implementation for optical–DSM semantic segmentation on the ISPRS Vaihingen dataset. The training entry point uses the C1 configuration: a shared frozen SAM 2 Hiera-S image encoder, modality-specific adapters, bidirectional fusion, diffusion prompts, and position-based cross-modal contrastive supervision.

## Setup

Use Python 3.10 or later and install a matching [PyTorch and torchvision build](https://pytorch.org/get-started/locally/) for your device. Then install the remaining dependencies:

~~~bash
pip install -r requirements.txt
~~~

Obtain the original SAM 2 Hiera-S checkpoint, sam2_hiera_small.pt, from [Meta's SAM 2 repository](https://github.com/facebookresearch/sam2). SAM 2 and SAM 2.1 checkpoints are distinct. The commands below take a local path to the original SAM 2 checkpoint.

## Data

Prepare spatially aligned 1024 × 1024 TIFF tiles from the [ISPRS Vaihingen benchmark](https://www.isprs.org/education/benchmarks/UrbanSemLab/semantic-labeling.aspx). Each split must have matching file stems across its three directories:

~~~text
Vaihingen-1024/
  train/{img,dsm,gray_label}/
  val/{img,dsm,gray_label}/
  test/{img,dsm,gray_label}/
~~~

Images are RGB and DSM tiles are single-channel. Grayscale labels store integer class IDs in the order given in [classes.json](vaihingen/utils/classes.json). IDs 0–4 are the five foreground classes, 5 is clutter/background, and 255 is ignored. The published training setup uses 136 training, 38 validation, and 36 test tiles. The loader uses the C1 training split's DSM normalization statistics in [common.py](vaihingen/experiments/scripts/common.py).

## Training and evaluation

Run the following from the repository root and replace the example paths with your own:

~~~bash
python vaihingen/train.py --root /path/to/Vaihingen-1024 --sam-checkpoint /path/to/sam2_hiera_small.pt --output runs/vaihingen-c1

python vaihingen/test.py --root /path/to/Vaihingen-1024 --sam-checkpoint /path/to/sam2_hiera_small.pt --checkpoint runs/vaihingen-c1/best.pth --output runs/vaihingen-c1/test
~~~

Training uses seed 42, batch size 2, a maximum of 100 epochs, and 10 DDIM steps. It selects best.pth by validation foreground mIoU and resumes from last.pth when it is present in the specified output directory. The release trainer disables autocast weight caching during training so that the no-gradient DDIM pass cannot cache detached weights for the supervised denoising pass. This correction can change the outcome of a new training run relative to the archival C1 run.

Testing defaults to five DDIM steps and fixed noise seed 42. Use --ddim-steps 10 for the archival C1 sampling budget. Testing writes per-image grayscale predictions, a confusion matrix, per-class IoU and F1, and summary metrics in percent. Macro metrics average the five foreground classes, while overall accuracy includes valid background pixels. Load checkpoints only from sources you trust.

The train and test commands accept --device cpu for CPU execution. CUDA runs use FP16 AMP for training and testing, while training validation runs in FP32.

## Source layout

- [C1 model construction](vaihingen/experiments/scripts/ablation_model.py)
- [Encoder, fusion, diffusion and decoding modules](vaihingen/model)
- [Training entry point](vaihingen/train.py)
- [Test entry point](vaihingen/test.py)

Dataset files and trained checkpoints are supplied separately. Adapted [Meta SAM 2](https://github.com/facebookresearch/sam2) components retain their [Apache 2.0 license](vaihingen/model/SAM2/LICENSE). The MiT components follow the [NVIDIA SegFormer license](third_party/SegFormer-LICENSE.txt). The diffusion prompt module builds on [GeoSAM](https://github.com/110-011/GeoSAM).
