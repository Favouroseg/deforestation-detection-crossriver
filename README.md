# Deforestation Detection — Cross River State, Nigeria

Deep learning-based pixel-level deforestation detection using Sentinel-2 multispectral satellite imagery and Hansen Global Forest Change data.

This project investigates semantic segmentation models for detecting forest loss in Cross River State, Nigeria. Three U-Net-based approaches were trained and evaluated using the same dataset split:

1. Vanilla U-Net
2. U-Net with a ResNet34 encoder
3. Improved ResNet34 U-Net using a combined Dice + BCE + Focal Loss

The best-performing model was the Improved ResNet34 U-Net, achieving a test IoU of **0.3312**.

---

## Project Overview

Deforestation monitoring is an important application of remote sensing and computer vision. Satellite imagery provides large-scale observations of land cover, while deep learning segmentation models can be used to identify areas associated with forest loss.

This project uses four Sentinel-2 spectral bands together with forest-loss information from the Hansen Global Forest Change dataset to train deep learning models for binary pixel-level deforestation segmentation.

The workflow covers:

- Satellite imagery preparation
- Forest-loss label preparation
- Image and mask alignment
- Patch generation
- Dataset splitting
- U-Net model training
- ResNet34 encoder experiments
- Focal-loss experimentation
- Model evaluation
- Prediction threshold optimization
- Final deforestation prediction visualization

---

## Study Area

**Cross River State, Nigeria**

The study focuses on Cross River State, an important forested region in southeastern Nigeria.

The project uses satellite imagery covering the study area and investigates whether deep learning segmentation can identify pixels associated with forest loss.

---

## Dataset

### Sentinel-2 Imagery

The project uses Sentinel-2 multispectral satellite imagery with four bands:

- B8 — Near Infrared (NIR)
- B4 — Red
- B3 — Green
- B2 — Blue

The imagery was processed into a four-channel input used by the segmentation models.

### Hansen Global Forest Change

Forest-loss labels were obtained from the Hansen Global Forest Change dataset.

The Hansen `lossyear` layer was converted into a binary segmentation mask:

```text
0 = No forest loss
1 = Forest loss
