# AeroVision: Turbine Blade Defect Detector (YOLO11 Segment)

This project provides a local, end-to-end computer vision pipeline and a premium web dashboard to detect turbine blades and identify defects (cracks and burns) on them using **Ultralytics YOLO11 Instance Segmentation**.

---

## 📁 Workspace Folder Structure

```
c:\Users\ADMIN\Desktop\Rolling/
├── dataset/                     # Local dataset directory
│   ├── data.yaml                # YOLO11 dataset configuration
│   ├── train/                   # Training split (images & labels)
│   ├── val/                     # Validation split
│   └── test/                    # Test split (optional)
├── static/                      # Web UI Frontend Assets
│   ├── index.html               # Sleek glassmorphic dashboard
│   ├── style.css                # Premium custom styling & print styles
│   ├── app.js                   # Drag & drop upload, dynamic dashboard tables
│   └── borescope_test.jpg       # Copied mock image for immediate UI demoing
├── train.py                     # Module 1: YOLO11 Segmentation training pipeline
├── predict.py                   # Module 2: CLI-based single-image inference and overlays
├── evaluate.py                  # Module 3: Validation accuracy metric evaluator
├── utils.py                     # Module 4: Spatial grouping, area calculations, mock processor
├── app.py                       # Module 5: FastAPI backend server serving static assets and API
├── requirements.txt             # Python packages
├── borescope_test.jpg           # Standalone test image in project root
└── README.md                    # This documentation file
```

---

## ⚙️ Setup and Dependencies

Install the requirements listed in the root directory:

```bash
pip install -r requirements.txt
```

---

## 🗃️ Dataset Preparation

The pipeline is set up for YOLO11 Instance Segmentation format (polygons).

1. In Roboflow, export your dataset in **YOLO11 Instance Segmentation** format.
2. Unzip your dataset inside the `dataset/` directory. Your folder structure should match:
   - `dataset/train/images/` and `dataset/train/labels/`
   - `dataset/val/images/` and `dataset/val/labels/`
   - `dataset/data.yaml`
3. Ensure class indexing in your dataset matches:
   - `0: turbine_blade` (annotated as boxes or boundary polygons)
   - `1: crack` (annotated as segmentation masks/polygons)
   - `2: burn` (annotated as segmentation masks/polygons)

---

## 🚀 Usage Guide

### 1. Training locally (`train.py`)
Run the training script to train the model on your dataset. By default, it uses GPU (`cuda:0`) if available, falling back to CPU.

```bash
# Train on your dataset (parameters are configurable)
python train.py --epochs 100 --batch 16 --imgsz 640
```
- **Weights**: Checkpoint files will be automatically saved under `runs/segment/train_turbine_defects/weights/`. The best weights (`best.pt`) are used by inference scripts automatically.

### 2. Running single-image inference via CLI (`predict.py`)
Run predictions on any borescope image from the command line:

```bash
# Basic usage (automatically looks for your best trained weights or falls back to demo mode)
python predict.py borescope_test.jpg --output output_annotated.jpg

# Specifying custom weights and custom confidence thresholds
python predict.py path/to/image.jpg --model runs/segment/train_turbine_defects/weights/best.pt --conf 0.35
```

### 3. Model Evaluation (`evaluate.py`)
Compute mAP, precision, and recall scores on your validation dataset:

```bash
python evaluate.py --split val
```

### 4. Running the Interactive Dashboard (`app.py`)
Launch the web interface dashboard locally:

```bash
python app.py
```
1. Open your browser and navigate to: **`http://localhost:8000/`**
2. **Mock/Demo Mode**: If you haven't trained a model yet, click **"Load Demo Image"** at the top right to explore the full dashboard capabilities (synthetically generated overlays and metrics).
3. **Real Inference**: Drag and drop or browse to upload any borescope image. If a trained model is found, the system runs YOLO11 segmentation and displays:
   - Original vs. annotated side-by-side images.
   - Turbine blade, crack, and burn colored segment masks.
   - Diagnostic metrics: blade counts, defect counts, area calculations, percentage of blade area compromised, and severity levels.
4. **Export Reports**: Fill in the metadata sidebar (Engine ID, Blade Row, Inspector, Date, Notes) and click **"Print Inspection Report"** to generate a clean, print-ready document which can be saved directly as a PDF!
