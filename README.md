# Fabric Defect Detection and Localisation using Deep Neural Networks

A high-accuracy deep neural network classification framework for fabric image defect detection (AITEX / `CO5420 Fabric` Dataset). This project aims to accurately distinguish between **Defect-Free (0)** and **Defective (1)** fabric samples with classification accuracy exceeding **95%**.

---

## 📌 Key Architectural Innovations

1. **Aspect Ratio Preservation ($4096 \times 256 \to 16 \times 256 \times 256$)**
   - Standard downsampling distorts fine weave patterns and hides microscopic defects.
   - Images are sliced into 16 non-overlapping $256 \times 256$ spatial patches to maintain fine-grained feature resolution.

2. **Patch Cosine Self-Similarity & Spatial Anomaly Vectorizing**
   - Pairwise cosine similarity matrices ($16 \times 16$) capture structural homogeneity. Defect-free samples exhibit uniform similarity ($S_{ij} \approx 1.0$), while defective samples manifest distinct low-similarity outliers.
   - Deep anomalous patch feature extraction (1984-dim) pinpoints local spatial anomalies across the fabric surface.

3. **Multi-Backbone Feature Transfer Learning**
   - Deep spatial representations are extracted using frozen PyTorch backbones: `ResNet-18`, `ResNet-34`, and `MobileNetV3`.
   - Feature fusion combines deep representations (`mean`, `std`, `max`, `min`, `global_vec`), Gray-Level Co-occurrence Matrix (GLCM) statistical metrics, and distance ratios.

4. **Data Augmentation with StratifiedGroupKFold**
   - Incorporates horizontal/vertical flip augmentations to triple training set size ($3\times$).
   - `StratifiedGroupKFold` guarantees that original images and their augmented variants remain strictly in the same fold, completely eliminating data leakage between train and validation splits.

5. **Multi-Model Ensembling & Test-Time Augmentation (TTA)**
   - Calibrated ensemble blending predictions across diverse classifiers: **ExtraTrees**, **RandomForest**, **GradientBoosting**, **Support Vector Machines (RBF-SVM)**, **MLP Classifier**, and **Logistic Regression**.
   - Test-Time Augmentation (3-view averaging) ensures robust test-set performance.

---

## 📁 Repository Structure

```
.
├── Fabric_Defect_Detection_Task2_version1.ipynb    # Baseline Deep Classifier & MIL Attention Pipeline
├── Fabric_Defect_Detection_Task2_final_version.ipynb # Final Production Pipeline (3x Augmentation, StratifiedGroupKFold, 6-Model Ensemble)
├── .gitignore                                       # Git ignore rules for dataset, caches, and weights
└── README.md                                        # Project documentation
```

### Notebook Overview

* **[`Fabric_Defect_Detection_Task2_version1.ipynb`](file:///c:/Users/Dell/Downloads/fabric2/Fabric_Defect_Detection_Task2_version1.ipynb)**
  - Implements the initial high-accuracy DNN classifier using Multiple Instance Learning (MIL) attention, multi-backbone feature extraction, 5-fold Stratified CV, and optimal threshold calibration.

* **[`Fabric_Defect_Detection_Task2_final_version.ipynb`](file:///c:/Users/Dell/Downloads/fabric2/Fabric_Defect_Detection_Task2_final_version.ipynb)**
  - Final production version incorporating $3\times$ training augmentation, `StratifiedGroupKFold` split verification, variance thresholding, SelectKBest feature selection ($k=300$), 3-view Test-Time Augmentation (TTA), and a weighted 6-model ensemble.

---

## 🛠️ Environment Requirements

To run the Jupyter notebooks, install the required Python packages:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn scikit-image albumentations Pillow
```

---

## 📂 Dataset Setup

The notebooks expect the dataset to be structured under `CO5420 Fabric/` as follows:

```
CO5420 Fabric/
├── Train/
│   ├── defect_free/     # PNG images of defect-free fabric
│   └── defective/       # PNG images of defective fabric
└── Test/                # PNG images for test set inference
```

> **Note:** The dataset directory `CO5420 Fabric/` is excluded from git tracking via `.gitignore`.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd fabric2
   ```

2. **Place the dataset** inside `CO5420 Fabric/` following the directory layout above.

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. Open and run **[`Fabric_Defect_Detection_Task2_final_version.ipynb`](file:///c:/Users/Dell/Downloads/fabric2/Fabric_Defect_Detection_Task2_final_version.ipynb)** for the complete pipeline execution and final results.
