
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
````

The final training experiment used the binary forest-loss mask.

### Image Patches

The Sentinel-2 imagery and corresponding binary masks were divided into:

* Patch size: **256 × 256 pixels**
* Total paired patches: **2,877**
* Image shape: **4 × 256 × 256**
* Mask shape: **256 × 256**

Patches containing less than 1% deforestation were excluded during patch generation.

---

## Dataset Split

A fixed random split was used for all three model experiments to ensure that the models could be compared using the same data partition.

| Dataset    | Number of Patches |
| ---------- | ----------------: |
| Training   |             2,013 |
| Validation |               431 |
| Test       |               433 |
| **Total**  |         **2,877** |

The split was generated using a fixed random seed:

```text
Seed = 42
```

This allows the experiments to be reproduced using the same dataset partition.

> Note: The split used in the final experiments was a fixed random split rather than a geographic/spatial holdout.

---

# Model Experiments

Three segmentation models were evaluated.

## 1. Vanilla U-Net

The first experiment used a standard Vanilla U-Net architecture as the baseline model.

The model was trained for:

* Epochs: **40**
* Batch size: **4**
* Initial learning rate: **0.0001**
* Optimizer: **Adam**

The Vanilla U-Net provides a baseline against which the ResNet34-based approaches can be compared.

---

## 2. ResNet34 U-Net

The second experiment replaced the standard U-Net encoder with a pretrained ResNet34 encoder.

Configuration:

* Architecture: **U-Net**
* Encoder: **ResNet34**
* Encoder weights: **ImageNet**
* Input channels: **4**
* Output classes: **1**
* Epochs: **40**
* Batch size: **4**
* Initial learning rate: **0.0001**
* Optimizer: **Adam**

This experiment investigates whether a pretrained ResNet34 encoder improves segmentation performance compared with the Vanilla U-Net.

---

## 3. Improved ResNet34 U-Net + Focal Loss

The final experiment used the ResNet34 U-Net architecture with an improved combined loss function designed to address class imbalance and improve segmentation performance.

The combined loss consists of:

```text
Combined Loss =
    Dice Loss
    + 0.5 × Binary Cross-Entropy Loss
    + 0.5 × Focal Loss
```

The model was trained using the same dataset split as the other two experiments.

The final model produced the strongest overall IoU and F1 score among the three tested models.

---

# Model Comparison

The final test results are:

| Model                                    |        IoU | Precision |     Recall |   F1 Score | Accuracy |
| ---------------------------------------- | ---------: | --------: | ---------: | ---------: | -------: |
| Vanilla U-Net                            |     0.3006 |    0.4107 |     0.5284 |     0.4622 |   0.8466 |
| ResNet34 U-Net                           |     0.3109 |    0.4348 |     0.5216 |     0.4743 |   0.8558 |
| **Improved ResNet34 U-Net + Focal Loss** | **0.3312** |    0.4184 | **0.6137** | **0.4976** |   0.8454 |

The Improved ResNet34 U-Net achieved the highest:

* **IoU**
* **Recall**
* **F1 Score**

The ResNet34 U-Net without the additional loss improvements achieved the highest precision and accuracy.

---

# Final Model Performance

The final selected model was:

**Improved ResNet34 U-Net + Focal Loss**

Final test performance:

| Metric    |     Result |
| --------- | ---------: |
| IoU       | **0.3312** |
| Precision | **0.4184** |
| Recall    | **0.6137** |
| F1 Score  | **0.4976** |
| Accuracy  | **0.8454** |

Compared with the Vanilla U-Net baseline:

| Metric    | Improvement |
| --------- | ----------: |
| IoU       |     +0.0306 |
| Precision |     +0.0077 |
| Recall    |     +0.0853 |
| F1 Score  |     +0.0354 |
| Accuracy  |     -0.0012 |

The largest improvement was observed in recall, which increased from **0.5284** to **0.6137**.

---

# Prediction Threshold Selection

The segmentation model produces a probability for each pixel. A threshold is then used to convert these probabilities into binary predictions.

Several thresholds were evaluated on the validation set:

| Threshold |        IoU |  Precision |     Recall |         F1 |
| --------: | ---------: | ---------: | ---------: | ---------: |
|      0.30 |     0.2436 |     0.2533 |     0.8634 |     0.3917 |
|      0.40 |     0.2807 |     0.3039 |     0.7863 |     0.4384 |
|      0.50 |     0.3080 |     0.3542 |     0.7026 |     0.4710 |
|  **0.60** | **0.3222** | **0.4054** | **0.6109** | **0.4874** |
|      0.70 |     0.3143 |     0.4852 |     0.4715 |     0.4783 |
|      0.80 |     0.2506 |     0.6001 |     0.3009 |     0.4008 |

A threshold of **0.60** was selected because it produced the highest validation IoU.

The selected threshold was then used for the final test evaluation.

### Final Test Result at Threshold 0.60

```text
IoU        : 0.3312
Precision  : 0.4184
Recall     : 0.6137
F1 Score   : 0.4976
Accuracy   : 0.8454
```

---

# Prediction Analysis

The final model was also tested on individual image patches to examine prediction variation.

The model produced different prediction probabilities and predicted deforestation percentages across image patches, demonstrating that the model responds differently to different satellite-image inputs.

Example prediction outputs include patches with:

* Very low predicted deforestation
* Moderate predicted deforestation
* Large areas predicted as deforestation

Prediction visualizations are included in the project results where available.

---

# Methodology

The overall workflow is:

```text
Sentinel-2 Satellite Imagery
            │
            ▼
      Image Preprocessing
            │
            ▼
Hansen Forest-Loss Data
            │
            ▼
      Binary Mask Creation
            │
            ▼
    Image/Mask Alignment
            │
            ▼
       256 × 256 Patches
            │
            ▼
       Fixed Data Split
      ┌─────┼─────┐
      ▼     ▼     ▼
    Train  Val   Test
      │
      ▼
 ┌───────────────────────┐
 │   Model Experiments   │
 ├───────────────────────┤
 │ Vanilla U-Net         │
 │ ResNet34 U-Net        │
 │ Improved ResNet34    │
 │ + Focal Loss          │
 └───────────────────────┘
            │
            ▼
      Validation Tuning
            │
            ▼
     Threshold = 0.60
            │
            ▼
       Final Test
            │
            ▼
    Deforestation Map
```

---

# Evaluation Metrics

The models were evaluated using:

### Intersection over Union (IoU)

IoU measures the overlap between predicted and ground-truth deforestation regions.

```text
IoU = Intersection / Union
```

### Precision

Precision measures the proportion of predicted deforestation pixels that were actually deforestation pixels.

### Recall

Recall measures the proportion of actual deforestation pixels correctly identified by the model.

### F1 Score

F1 provides a balance between precision and recall.

### Accuracy

Accuracy measures the proportion of all pixels that were classified correctly.

Because the task involves pixel-level segmentation, IoU, precision, recall, and F1 are particularly important for understanding model behavior.

---

# Project Structure

The repository is organized around the notebook, model experiments, and generated results.

```text
deforestation-detection-crossriver/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── deforestation_detection.ipynb
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   ├── predictions/
│   └── metrics/
│
└── src/
    └── README.md
```

Large datasets, satellite GeoTIFF files, patch collections, and model checkpoints may not be included directly in the GitHub repository because of their file sizes.

---

# Technologies

The project uses:

* Python
* PyTorch
* Segmentation Models PyTorch
* Rasterio
* NumPy
* Matplotlib
* Google Colab
* CUDA
* Google Drive
* Google Earth Engine
* Sentinel-2
* Hansen Global Forest Change

---

# Hardware

Model training was performed using Google Colab with an NVIDIA Tesla T4 GPU.

```text
GPU: NVIDIA Tesla T4
Framework: PyTorch
CUDA acceleration: Enabled
```

---

# Reproducibility

The experiments use a fixed random seed:

```python
SEED = 42
```

The same train/validation/test split was used across the three model experiments.

To reproduce the experiments:

1. Open the project notebook in Google Colab.
2. Mount Google Drive.
3. Prepare the Sentinel-2 imagery and Hansen forest-loss data.
4. Generate the image and mask patches.
5. Create the fixed dataset split.
6. Train the Vanilla U-Net.
7. Train the ResNet34 U-Net.
8. Train the Improved ResNet34 U-Net.
9. Evaluate the models on the test set.
10. Perform validation threshold analysis.
11. Use the selected threshold of 0.60 for the final evaluation.

---

# Limitations

Several limitations should be considered when interpreting the results.

### Dataset and Labels

The model relies on Hansen forest-loss data as the reference labels. Differences between the satellite imagery and the forest-loss product may introduce uncertainty into the training labels.

### Spatial Generalization

The current train/validation/test split is a fixed random split. It should not be interpreted as a geographic spatial holdout. Future work could evaluate the model using geographically separated regions to better measure spatial generalization.

### Model Performance

The final IoU of **0.3312** indicates that the model is able to detect meaningful deforestation patterns, but there is still substantial room for improvement.

### Class Imbalance

Deforestation occupies a smaller proportion of the imagery than non-deforested areas. This imbalance motivated experimentation with Dice and Focal Loss.

### Resolution

The spatial resolution of Sentinel-2 imagery limits the detection of very small-scale forest disturbances.

---

# Future Work

Potential improvements include:

* Spatially separated train/validation/test splits
* Additional Sentinel-2 spectral bands
* Multi-temporal Sentinel-2 imagery
* Data augmentation
* Higher-resolution training data
* More advanced segmentation architectures
* Hyperparameter optimization
* Improved handling of class imbalance
* Post-processing of predicted deforestation regions
* Evaluation against independently verified deforestation data
* Testing the model on other forest regions in Nigeria

---

# Results Summary

The experiments show that incorporating a pretrained ResNet34 encoder improved performance over the Vanilla U-Net, while the addition of the combined Dice + BCE + Focal Loss further improved the final model.

The final model achieved:

> **IoU: 0.3312**
> **Precision: 0.4184**
> **Recall: 0.6137**
> **F1 Score: 0.4976**
> **Accuracy: 0.8454**

The prediction threshold was selected using the validation dataset, with **0.60** producing the highest validation IoU.

---

# Author

**Favour Oseghale**

Information and Communication Engineering Student
Covenant University, Nigeria

GitHub: [@Favouroseg](https://github.com/Favouroseg)

---

```
```
