# NeuroScan AI: Brain Tumor Detection & Clinical Decision Suite

An end-to-end deep learning diagnostic workstation for brain tumor classification from cranial MRI scans. Powered by a transfer-learned **VGG-16** convolutional neural network, with model explainability via **Grad-CAM**, native **DICOM (`.dcm`)** support, longitudinal scan comparison, and patient care guidance.

---

## 🚀 Key Features

- **Multiclass Neoplasm Detection**: Accurately classifies cranial scans into four categories:
  - **Glioma**
  - **Meningioma**
  - **Pituitary Adenoma**
  - **No Tumor (Healthy / Normal)**
- **Grad-CAM Visual Heatmaps**: Uses gradient-weighted class activation mapping at the final convolutional layer (`block5_conv3`) to visually pinpoint the exact region of the lesion on the scan.
- **Hospital DICOM (`.dcm`) Support**: Directly ingest 16-bit hospital scanner files, extracting clinical metadata (Patient Age, Sex, Field Strength, Slice Thickness) and normalizing intensity windowing for inference.
- **Out-of-Distribution Image Validator**: Automatically detects and rejects non-cranial images (photos, pets, documents, blank scans) with informative feedback.
- **Longitudinal Scan Comparison**: Side-by-side comparative analysis of **Baseline (Pre-treatment)** vs **Follow-up (Post-treatment)** scans to compute lesion delta and remission/progression status.
- **Clinical Severity & Patient Care Advisory**:
  - Threat & urgency levels with recommended specialist action windows.
  - Recommended dietary nutrition (anti-inflammatory foods to favor vs simple sugars/sodium to avoid).
  - Emergency red-flag symptom warnings.
- **Printable Medical Diagnostic Report**: Export formatted hospital clinical summaries ready for print or PDF save, complete with side-by-side Grad-CAM heatmaps and radiologist sign-off blocks.
- **Persistent Case History**: Lightweight SQLite backend storing past examinations with 1-click historical reload.

---

## 🧠 Model Architecture

- **Base Architecture**: VGG-16 (Pre-trained on ImageNet, fine-tuned top convolutional blocks).
- **Classifier Head**:
  - Flatten layer
  - Dropout (0.3)
  - Dense (128 units, ReLU activation)
  - Dropout (0.2)
  - Dense (4 units, Softmax activation)
- **Input Dimensions**: 128 x 128 x 3 (RGB normalized tensors)
- **Optimizer**: Adam (`lr=0.0001`) with Sparse Categorical Crossentropy

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Git LFS (for Model Weights)
The trained model weights (`model.h5`, ~122MB) are managed with Git Large File Storage (Git LFS).
```bash
git lfs install
git lfs pull
```

### 5. Run the Application
```bash
python main.py
```
Open your browser and navigate to **http://127.0.0.1:5000**.

---

## 📂 Project Structure

```text
├── main.py                  # Flask web backend & REST API endpoints
├── mri_utils.py             # Grad-CAM generator, DICOM processor & scan validator
├── db.py                    # SQLite database interface for case history
├── model.h5                 # Trained VGG-16 deep learning weights
├── templates/
│   └── index.html           # Diagnostic workstation interface
├── uploads/                 # Storage for uploads & generated Grad-CAM heatmaps
│   ├── Te-gl_9.jpg          # Demo Glioma scan
│   ├── Te-aug-me_25.jpg     # Demo Meningioma scan
│   ├── Te-pi_28.jpg         # Demo Pituitary scan
│   ├── Te-no_43.jpg         # Demo Normal scan
│   └── test_synth.dcm       # Demo DICOM scan
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation
```

---

## 📋 Clinical Disclaimer
NeuroScan AI is intended for academic research and clinical decision support purposes. Artificial intelligence models should always be interpreted in conjunction with comprehensive clinical history and confirmed by a board-certified neuroradiologist.
