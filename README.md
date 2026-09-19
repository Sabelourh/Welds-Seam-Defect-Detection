# Automated Weld Seam Defect Detection

End-to-end academic computer-vision project for Sabelo Ndlovu.

## What is included
- Real-dataset downloader for the CC0 Kaggle **Weld Quality Inspection - Instance Segmentation** dataset.
- Conversion of YOLO annotations to a conservative image-classification dataset.
- Dataset analysis: class counts, imbalance chart, sample grid and CSV.
- ResNet18 transfer-learning classifier.
- Best-model checkpoint saved as `models/best_weld_model.pth`.
- Test evaluation: accuracy, precision, recall, F1, classification report and confusion matrix.
- Training curves: loss and accuracy.
- OpenCV-vs-CNN comparison on the same classification test set.
- Mobile-friendly Gradio app with upload/webcam support and CNN confidence scores.
- OpenCV fallback when no trained checkpoint exists.

## Important label decision
The selected public dataset documents these labels:
`Bad Welding`, `Crack`, `Excess Reinforcement`, `Good Welding`, `Porosity`, `Spatters`.

This project trains the requested classification baseline on:
`Good Welding`, `Porosity`, `Crack`, `Spatters`.

**Undercut is not claimed**, because this dataset does not document an Undercut class.

Images carrying multiple target defect labels are excluded from the image-level classifier so that a single-label ResNet18 target is not fabricated. The original YOLO annotations remain in `data/raw`.

## Quick start

### 1. Create environment
Recommended: Python 3.11 or 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the real dataset
```bash
python src/download_dataset.py
```

If Kaggle requires authentication on your machine, sign in/configure Kaggle and rerun. You can also manually download the dataset and extract it under `data/raw`.

### 3. Prepare classification folders
```bash
python src/prepare_dataset.py
```

### 4. Analyse dataset
```bash
python src/analyze_dataset.py
```

### 5. Train CNN
```bash
python src/train_cnn.py --epochs 20
```

### 6. Evaluate
```bash
python src/evaluate_cnn.py
```

### 7. Compare OpenCV and CNN
```bash
python src/compare_opencv_cnn.py
```

### 8. Launch website
```bash
python app.py
```

## One-command pipeline
```bash
python run_pipeline.py
```

## Outputs
After a real training run:
- `models/best_weld_model.pth`
- `results/dataset_counts.csv`
- `results/class_distribution.png`
- `results/sample_images.png`
- `results/training_history.csv`
- `results/training_accuracy.png`
- `results/training_loss.png`
- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/metrics.json`
- `results/opencv_vs_cnn.csv`
- `results/opencv_vs_cnn_summary.json`

## Scientific integrity
No accuracy, F1 score, confusion matrix, training curve, or trained checkpoint is pre-fabricated. They are generated only after the real dataset has been downloaded and the training/evaluation scripts have actually run.

The application is an academic prototype, not a certified NDT inspection system.
