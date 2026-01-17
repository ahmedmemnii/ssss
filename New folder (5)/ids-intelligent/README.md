# Intelligent Intrusion Detection System (IDS)

This project builds a machine-learning and deep-learning based IDS capable of detecting anomalous network behaviors, comparing multiple models, and visualizing alerts.

## Features
- Data ingestion (demo: KDDCup99) and preprocessing pipeline
- Supervised models: RandomForest, SVM, KNN
- Unsupervised models: IsolationForest, K-Means
- Deep learning: PyTorch Autoencoder for anomaly detection
- Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC
- Streamlit app: visualize alerts, anomaly scores, and traffic charts
- Saved models (`models/`) and comparison table (`models/performance.csv`)

## Quick Start

### 1) Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train & Evaluate (demo pipeline)
```bash
python src/ids/train.py --dataset kddcup99 --models rf svm knn iso kmeans ae --save
```

### 3) Run the Streamlit App
```bash
streamlit run app/app.py
```

### 4) Use a Custom CSV
Place your CSV under `data/raw/` and run:
```bash
python src/ids/train.py --csv data/raw/your_data.csv --label-column label --save
```

## Structure
- `src/ids/` — core library (data, preprocess, models, train, evaluate)
- `app/` — Streamlit UI for alerts and anomaly scores
- `notebooks/` — exploratory Jupyter notebooks
- `reports/` — LaTeX report
- `data/` — raw and processed datasets
- `models/` — saved models and performance outputs

## Notes
- For production, integrate with SIEM via connectors or file-based ingestion; consider MLOps for retraining.
- For CICIDS2017/UNSW-NB15, adapt `src/ids/data.py` to parse their CSV schemas.
