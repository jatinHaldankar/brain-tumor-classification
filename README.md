# Brain Tumor MRI Classification

End-to-End Deep Learning project for **Brain Tumor MRI Classification** (4 classes: Glioma, Meningioma, No Tumor, Pituitary) using VGG16, MLflow, and DVC.

## About The Project

Brain tumors are severe and life-threatening conditions that require early and accurate detection for effective treatment planning. Magnetic Resonance Imaging (MRI) is the standard method used by radiologists to diagnose these tumors. However, manually analyzing these scans is a highly time-consuming process that can be prone to human error due to fatigue or subtle visual differences.

**The Problem:** Radiologists face an overwhelming volume of MRI scans daily. Rapidly and accurately distinguishing between different types of brain tumors is critical for patient survival, but manual classification creates a significant bottleneck in the diagnostic pipeline.

**The Solution:** This project provides an automated, End-to-End Deep Learning solution to classify brain tumors into four distinct categories: **Glioma, Meningioma, Pituitary, and No Tumor**. By leveraging a fine-tuned VGG16 Convolutional Neural Network (CNN), this application acts as an assistive tool for medical professionals, significantly accelerating the diagnostic process and providing a reliable, data-driven second opinion.

This project is built using production-grade MLOps practices, featuring automated data versioning (DVC), experiment tracking (DagsHub/MLflow), and a full Continuous Deployment (CI/CD) pipeline that serves the model via an interactive web interface on Azure Container Apps.

## Project Structure

```
├── config/config.yaml        # Pipeline configuration
├── params.yaml               # Model hyperparameters
├── src/cnnClassifier/
│   ├── components/           # Data ingestion, model training, evaluation
│   ├── pipeline/             # Stage-wise pipeline scripts
│   ├── config/               # Configuration manager
│   ├── entity/               # Dataclass config entities
│   └── utils/                # Common utilities
├── research/                 # Jupyter notebooks (exploratory)
├── dvc.yaml                  # DVC pipeline definition
└── main.py                   # Entry point
```

## Workflows

1. Update `config/config.yaml`
2. Update `params.yaml`
3. Update the entity (`config_entity.py`)
4. Update the configuration manager
5. Update the components
6. Update the pipeline scripts
7. Update `dvc.yaml`

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/jatinHaldankar/brain-tumor-classification
cd brain-tumor-classification
```

### 2. Create virtual environment
```bash
pip install uv
uv venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
uv pip install -e .
```

### 4. Configure credentials

Create a `.env` file in the project root:
```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
MLFLOW_TRACKING_URI=https://dagshub.com/jatinHaldankar/brain-tumor-classification.mlflow
MLFLOW_TRACKING_USERNAME=jatinHaldankar
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 5. Run the pipeline
```bash
dvc repro
```

## MLflow Tracking

Experiments are tracked on DagsHub:
👉 https://dagshub.com/jatinHaldankar/brain-tumor-classification.mlflow

To view locally:
```bash
mlflow ui
```

## DVC Commands

```bash
dvc repro        # Run the full pipeline
dvc dag          # View the pipeline DAG
dvc metrics show # View evaluation metrics
```

## Model

- **Architecture:** VGG16 (pretrained on ImageNet) + custom classifier head
- **Fine-tuning:** Last convolutional block (block5) unfrozen
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  - Training: 5,600 images (1,400 per class)
  - Testing: 1,600 images (400 per class)

## Results

| Metric   | Score |
|----------|-------|
| Accuracy | 0.889 |
| Loss     | 0.470 |

Tracked via MLflow on DagsHub.