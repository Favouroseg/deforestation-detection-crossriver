# deforestation-detection-crossriver
# Deforestation Detection — Cross River State, Nigeria

An AI-powered pipeline that detects and maps deforestation in Cross River State, Nigeria — home to Nigeria's largest remaining rainforest — using Sentinel-2 satellite imagery and a U-Net deep learning model. The model outperforms a Random Forest baseline across all evaluation metrics.

---

## What This Project Does

This project pulls real satellite imagery of Cross River State from Google Earth Engine, processes it into training patches paired with ground truth forest loss labels from the Hansen Global Forest Change dataset, and trains a U-Net segmentation model to detect deforestation at the pixel level. The result is a model that can look at a satellite image patch and predict exactly which pixels represent deforested land.

---

## Results

| Metric | Random Forest (Baseline) | U-Net (Ours) |
|--------|--------------------------|--------------|
| IoU | 0.1582 | **0.2764** |
| Accuracy | 0.5130 | **0.7266** |
| Precision | 0.1717 | **0.2592** |
| Recall | 0.6684 | **0.8033** |
| F1 Score | 0.2732 | **0.3919** |

The U-Net achieves a **recall of 0.8033** — meaning it correctly identifies 80% of all actual deforestation pixels. For conservation monitoring, recall matters most: missing real deforestation is worse than a false alarm.

---

## Study Area

**Cross River State, Nigeria** — Nigeria's last major contiguous rainforest and a documented hotspot for illegal logging and agricultural encroachment. Despite its ecological significance, the region is critically understudied in the remote sensing literature. This project is, to our knowledge, the first deep learning study specifically targeting deforestation detection in this area.

---

## Dataset

- **Satellite imagery:** Sentinel-2 multispectral imagery (4 bands: B8, B4, B3, B2) pulled via Google Earth Engine, 2021–2023 median composite
- **Labels:** Hansen Global Forest Change dataset (v1.13, 2000–2025), binarized to forest loss / no loss
- **Patches:** 2,877 paired image/mask patches (256×256 pixels), filtered to include only patches with >1% deforestation
- **Split:** 70% train / 15% validation / 15% test (spatial holdout)

---

## Model Architecture

- **Architecture:** U-Net with ResNet34 encoder
- **Pretrained weights:** ImageNet (transfer learning)
- **Input channels:** 4 (Sentinel-2 multispectral)
- **Output:** Binary segmentation mask (forest loss / intact forest)
- **Loss function:** Combined Dice Loss + Binary Cross-Entropy (BCE) — the combination was key to stabilizing IoU on the imbalanced dataset (6.4% deforestation vs 93.6% intact forest)
- **Optimizer:** Adam (lr=0.001)
- **Trained on:** Google Colab T4 GPU

---

## Stack

- Python
- Google Earth Engine + geemap
- PyTorch + segmentation-models-pytorch
- Rasterio
- Google Colab (GPU)

---

## Project Structure

```
deforestation_project/
├── data/
│   ├── raw_imagery/          # Sentinel-2 tiles
│   └── labels/               # Hansen lossyear
├── patches/
│   ├── images/               # 256×256 image patches (.npy)
│   └── masks/                # Corresponding binary masks (.npy)
├── models/
│   └── best_model.pth        # Best saved model checkpoint
└── outputs/
    └── maps/                 # Deforestation prediction maps
```

---

## How to Run

1. Open `deforestation_detection.ipynb` in Google Colab
2. Mount Google Drive and run the setup cell
3. Authenticate Google Earth Engine with your project ID
4. Run cells in order — data pull → preprocessing → training → evaluation

---



## Author

**Favour Oseghale** — Information and Communication Engineering student, Covenant University Nigeria | NCAIR Intern   
GitHub: [@Favouroseg](https://github.com/Favouroseg) | LinkedIn: [favouroseghale](https://linkedin.com/in/favouroseghale)

---

