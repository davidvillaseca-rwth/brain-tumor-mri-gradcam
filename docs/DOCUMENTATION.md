# Detailed documentation

This document explains every step of the project: the data, how the model is trained, how the heatmaps are made, how the glioma problem was found and fixed, and how to reproduce every number in the [README](../README.md).

**Contents**

1. [Dataset](#1-dataset)
2. [Model and training](#2-model-and-training)
3. [Grad-CAM](#3-grad-cam)
4. [The glioma problem: a shortcut](#4-the-glioma-problem-a-shortcut)
5. [Results](#5-results)
6. [Reproducing everything](#6-reproducing-everything)
7. [Code reference](#7-code-reference)
8. [Limitations and next steps](#8-limitations-and-next-steps)

---

## 1. Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle. It contains 7,200 JPEG images of single 2-D MRI slices, already split into folders:

| Class | Training | Testing |
|---|---|---|
| glioma | 1,400 | 400 |
| meningioma | 1,400 | 400 |
| no tumor | 1,400 | 400 |
| pituitary | 1,400 | 400 |
| **Total** | **5,600** | **1,600** |

The dataset is a merge of three public collections (figshare, SARTAJ and Br35H). It includes axial, coronal and sagittal views and several MRI sequences, and image sizes range from about 200 to 1,400 pixels. This mix of sources turns out to matter a lot (section 4).

`download_data.py` uses `kagglehub` to fetch it into `data/`:

```
data/
├── Training/{glioma,meningioma,notumor,pituitary}/*.jpg
└── Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
```

### Splits

| Split | Source | Size | Used for |
|---|---|---|---|
| train | 85% of `Training/` (fixed seed 42) | 4,760 | learning weights |
| val | other 15% of `Training/` | 840 | choosing the best epoch |
| test | `Testing/` | 1,600 (1,562 for the final model) | the final score, never used to train or pick epochs |

The final model also moves 38 images from `Testing/` into training. Section 4 explains why and how.

---

## 2. Model and training

### Transfer learning

Training a deep network from scratch on 5,000 images would overfit. Instead, `train.py` starts from **ResNet50 with ImageNet weights** (`IMAGENET1K_V2` in torchvision). Those weights already detect edges, textures and shapes. Only the last layer is replaced, by a new `Linear(2048 → 4)` for the four classes. Then the **whole network** is fine-tuned at a small learning rate, so the pretrained features adapt to MRI images.

### Preprocessing

Every image is converted to RGB, resized to 224 × 224 and normalised with the ImageNet mean and standard deviation, because that's what the pretrained weights expect.

### Augmentation

Augmentation randomly changes the training images, so the model has to learn what stays the same (the tumor) rather than incidental details.

| Transform | Light (original model) | Strong (`--strong-aug`) |
|---|---|---|
| Crop | none | `RandomResizedCrop`, scale 0.6–1.0 |
| Low resolution | none | 50% chance: downscale to 96–224 px, then upscale again |
| Horizontal flip | yes | yes |
| Rotation | ±10° | ±15° |
| Brightness / contrast | ±20% | ±40% |
| Gaussian blur | none | 30% chance, σ 0.1–2.0 |

The strong version is designed to break the source shortcut: crops, blur and low resolution hide the sharpness and framing differences between collections.

### Hyperparameters

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | One-cycle schedule, peak 3e-4 |
| Loss | Cross-entropy with label smoothing 0.05 |
| Batch size | 32 |
| Epochs | 6 (original), 10 (strong aug and final) |
| Precision | Mixed (`torch.autocast` + `GradScaler`) |
| Checkpoint | Best validation accuracy |
| Hardware | NVIDIA RTX 3060 Laptop GPU (6 GB), about 28 s per epoch |

Label smoothing caps predicted confidence at around 95–98%. So 96% confidence in the heatmaps means "as sure as this model gets", not literally 96% likely.

The checkpoint stores the weights and the class order (`{"state_dict": ..., "classes": [...]}`), so the other scripts never hardcode which index is which class.

---

## 3. Grad-CAM

Grad-CAM ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)) answers the question *"which pixels made the model choose this class?"* `gradcam.py` implements it in about 15 lines, with no extra library:

1. A **forward hook** on `model.layer4[-1]`, the last convolutional block, saves its output `A`: 2,048 feature maps, each 7 × 7.
2. The score of the predicted class, `y_c`, is differentiated with respect to those maps: `∂y_c / ∂A`.
3. Averaging each gradient map gives one **importance weight per channel**, `α_k`.
4. The heatmap is `ReLU(Σ_k α_k · A_k)`, a weighted sum of the feature maps. The ReLU keeps only evidence *for* the class.
5. The 7 × 7 map is upsampled to 224 × 224, scaled to 0–1 and drawn over the image with a `jet` colormap.

The model stays in `eval()` mode, but gradients stay on. `torch.autograd.grad` is used instead of backward hooks, which can misbehave with ResNet's in-place ReLUs.

How to read the maps: red means the region pushed the prediction strongly. Because the feature map is only 7 × 7, the heatmaps are blobby; they point to a region, not to exact tumor borders.

---

## 4. The glioma problem: a shortcut

### Symptom

The first model (6 epochs, light augmentation) scored **95.4%** on the full test set, but glioma recall was only **83.3%**. 36 gliomas were predicted as "no tumor", the most dangerous kind of mistake. The odd part: validation accuracy was 99%. A gap that large between validation and test usually means the test set differs from the training set in some way.

### Diagnosis

Comparing image sizes per class (`python analyze_sources.py`) gave the answer:

```
Share of images that are exactly 512x512 (one source):
  Training glioma      1400/1400  (100%)
  Training meningioma  1267/1400  (90%)
  Training notumor       17/1400  (1%)
  Training pituitary   1328/1400  (95%)
  Testing  glioma       324/400   (81%)
  ...
Test gliomas, predictions of the original model:
  512x512     n=324  correct 98.5%
  other size  n=76   correct 18.4%   meningioma:26  notumor:36
```

Image size works as a fingerprint for the source collection. In training, **every glioma comes from the 512 × 512 source, and almost every "no tumor" scan comes from other sources**. The model found a cheap rule: *"doesn't look like the 512 source → no tumor."* That works perfectly on training and validation data. It fails on the 76 test gliomas from a different collection.

To rule out bad labels, I looked at the other-source gliomas that the original model called "no tumor" (`python gradcam.py data/Testing/glioma/Te-gl_129.jpg ...` shows them with heatmaps). The tumors were plainly visible in every one, so the labels are fine. The model had learned the wrong thing.

This is a textbook case of **shortcut learning**: a model takes a feature that happens to correlate with the label in the dataset (here, the hospital or collection) instead of the feature that actually matters.

### Fix 1: strong augmentation

`--strong-aug` (section 2) makes source cues such as sharpness, resolution and framing unreliable. Without changing the data, other-source glioma recall rose from 23.7% to 55.3%.

### Fix 2: show the model gliomas from other sources

Augmentation can't create examples that aren't there. The model had **never seen a glioma from another source**. `--borrow` fixes that honestly:

1. Find the 76 test gliomas that are not 512 × 512.
2. Group near-duplicates using an 8 × 8 average image hash. The test folder contains copies such as `Te-gl_204` and `Te-gl_259`, and a copy on each side would leak test data into training.
3. Shuffle the groups with a fixed seed and move about half (**38 images**) into training, each **repeated 5 times** so that 38 images among 4,760 still count. They get the same augmentation as the rest.
4. Remove those 38 from the test set. Every model is then scored on the same **remaining 1,562 images**, using `compare.py`.

Result: other-source glioma recall rose to **76.3%**, and gliomas missed as "no tumor" fell from 20 to 4 on the shared test set.

**Trade-off:** the model now says "glioma" slightly more readily. 9 meningiomas are predicted as glioma, so meningioma recall dropped from 98.3% to 96.8%. Mixing up two tumor types is a much milder error than missing a tumor.

---

## 5. Results

### All models on the shared test set (1,562 images)

| Model | Command | Accuracy | Glioma recall | Other-source glioma | Glioma → no tumor |
|---|---|---|---|---|---|
| Original | `train.py --epochs 6` | 97.4% | 90.6% | 23.7% (9/38) | 20 |
| Strong aug | `train.py --strong-aug --epochs 10` | 97.9% | 94.5% | 55.3% (21/38) | 8 |
| **Final** | `train.py --strong-aug --borrow --epochs 10` | **98.0%** | **96.4%** | **76.3% (29/38)** | **4** |

On the full 1,600-image test set, the original model scored 95.4% and the strong-aug model 96.3%. The shared-set numbers are higher because the 38 borrowed images were the hardest ones.

### Final model, confusion matrix (1,562 images)

|  | glioma | meningioma | no tumor | pituitary | recall |
|---|---|---|---|---|---|
| **glioma** | 349 | 8 | 4 | 1 | 96.4% |
| **meningioma** | 9 | 387 | 0 | 4 | 96.8% |
| **no tumor** | 1 | 0 | 399 | 0 | 99.8% |
| **pituitary** | 2 | 2 | 0 | 396 | 99.0% |

### Heatmaps before and after

Original model:

![Grad-CAM, original model](images/gradcam_original.png)

Final model:

![Grad-CAM, final model](images/gradcam_best.png)

Both grids use the same random seed: 3 correct scans per class plus the first 4 mistakes found. Test gliomas that were borrowed into training are left out.

---

## 6. Reproducing everything

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python download_data.py

# the three models (about 3, 5 and 5 minutes on an RTX 3060 laptop)
python train.py --epochs 6                                   # -> models/resnet50_brain_mri.pt
python train.py --strong-aug --epochs 10 --out models/resnet50_strongaug.pt
python train.py --strong-aug --borrow --epochs 10 --out models/resnet50_borrow.pt

# the analysis and figures
python analyze_sources.py
python compare.py
python gradcam.py --model models/resnet50_brain_mri.pt --out docs/images/gradcam_original.png
python gradcam.py --out docs/images/gradcam_best.png
python docs/make_results_chart.py
```

Seeds fix the data split and the borrowed images. GPU training isn't bit-for-bit deterministic, so retrained models will land within a few tenths of a percent of the numbers above. Run `python train.py --smoke` first for a 30-second check that everything is installed.

The trained final model is attached to the [GitHub release](https://github.com/davidvillaseca-rwth/brain-tumor-mri-gradcam/releases) if you only want to run `gradcam.py`.

---

## 7. Code reference

### `train.py`

| Flag | Default | Meaning |
|---|---|---|
| `--epochs N` | 6 | Number of training epochs |
| `--batch N` | 32 | Batch size |
| `--strong-aug` | off | Use the strong augmentation from section 2 |
| `--borrow` | off | Move half of the other-source test gliomas into training (section 4) |
| `--out PATH` | `models/resnet50_brain_mri.pt` | Where to save the best checkpoint |
| `--smoke` | off | 128 training images and 1 epoch, to check the setup |

It prints per-epoch loss, train and validation accuracy, then the test accuracy and confusion matrix of the best checkpoint.

Useful functions: `build_model(n)` returns ResNet50 with an `n`-class head; `borrow_split(test_ds)` returns the borrowed and kept test indices.

### `gradcam.py`

```bash
python gradcam.py [scans ...] [--model PATH] [--out PATH]
```

Without scans, it draws a grid of unseen test scans to `gradcam_grid.png`. With scans, it draws your images to `gradcam_custom.png`. The `GradCAM` class can be reused: `cls, confidence, heatmap = GradCAM(model)(tensor)`.

### `compare.py`

Scores the three checkpoints (whichever exist in `models/`) on the shared 1,562-image test set: overall accuracy, glioma recall, other-source glioma recall and glioma → no tumor count.

### `analyze_sources.py`

```bash
python analyze_sources.py [--model PATH]
```

Prints the share of 512 × 512 images per split and class, and glioma recall by source group for the chosen model.

---

## 8. Limitations and next steps

**Limitations**

- **Not a medical device.** This is an educational project. It hasn't been validated clinically and must not be used to diagnose anyone.
- **2-D slices only.** Radiologists read the full 3-D volume, several sequences and the patient's history. A single slice loses most of that.
- **An easy benchmark.** A few public collections and near-duplicate images make this dataset easier than real hospital data. Expect lower accuracy on scans from a new scanner or hospital, exactly the kind of shift section 4 shows.
- **A reused test set.** The test set was evaluated several times while the fix was being developed, and the other-source glioma group is small (38 images, so each one is 2.6 percentage points). The direction of the improvement is clear; the exact figure is not.
- **Coarse heatmaps.** Grad-CAM at 7 × 7 resolution shows roughly where the model looks, not tumor boundaries. A sensible heatmap doesn't prove the model reasons the way a doctor does.
- **No CT.** At least one test "MRI" is actually a CT scan, and the model doesn't handle it.

**Next steps**

1. **External validation:** test on a separate public dataset that none of this data comes from. It's the most convincing check of generalisation.
2. **Remove duplicates across the whole dataset** (training vs test) with image hashing, and report the clean score.
3. **More sources in training:** add gliomas from other collections instead of borrowing from the test set.
4. **A small demo app** (for example Gradio): upload a scan, see the prediction and the heatmap.
5. **Calibration and uncertainty:** temperature scaling, or flagging low-confidence scans for a human to review.
