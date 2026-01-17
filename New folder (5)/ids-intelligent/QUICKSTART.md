# 🚀 QUICK START GUIDE - IDS Intelligent

## Installation (5 minutes)

### 1. Navigate to Project
```bash
cd "C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent"
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Verify Installation
```bash
python verify_setup.bat
```

---

## Training Models (10-30 minutes depending on models)

### Quick Training (Random Forest only - 5 min)
```bash
python -m src.ids.train --dataset kddcup99 --models rf --save
```

### Full Training (All models - 20-30 min)
```bash
python -m src.ids.train --dataset kddcup99 --models rf svm knn iso kmeans ae --save
```

**Output:**
- Trained models → `models/model_*.pkl` or `models/model_*.pt`
- Performance table → `models/performance.csv`
- Console shows accuracy, precision, recall, F1, confusion matrices

---

## Launching Dashboard (2 minutes)

### Start Streamlit App
```bash
streamlit run app/app.py
```

**Opens in browser:** http://localhost:8501

**Features:**
- Select trained model from dropdown
- Upload CSV with network traffic features
- View predictions (normal vs attack)
- See summary metrics and anomaly chart

---

## Generating Report PDF

### Compile LaTeX
```bash
cd reports
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

**Output:** `reports/report.pdf`

---

## Custom Dataset Usage

### Prepare Your CSV
Must have:
- Feature columns (numeric/categorical)
- Label column (e.g., "label", "attack_type")

### Train on Custom Data
```bash
python -m src.ids.train --csv data/raw/your_data.csv --label-column label --models rf svm --save
```

---

## Project File Reference

| File/Folder | Purpose |
|-------------|---------|
| `src/ids/data.py` | Load datasets (KDDCup99, CSV) |
| `src/ids/preprocess.py` | Feature engineering, splitting |
| `src/ids/models.py` | ML/DL model definitions |
| `src/ids/train.py` | Training pipeline (CLI) |
| `src/ids/evaluate.py` | Metrics calculation |
| `app/app.py` | Streamlit dashboard |
| `notebooks/ids_exploration.ipynb` | Interactive analysis |
| `reports/report.tex` | Technical documentation |
| `models/` | Saved models & performance |
| `data/raw/` | Original datasets |
| `data/processed/` | Cleaned data |

---

## Common Commands

### View Model Performance
```bash
python -c "import pandas as pd; print(pd.read_csv('models/performance.csv'))"
```

### Test Model in Python
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/model_rf.pkl')

# Predict on new data
df = pd.read_csv('data/raw/test_traffic.csv')
predictions = model.predict(df)
print(predictions)
```

### Export Predictions to CSV
```python
import joblib
import pandas as pd

model = joblib.load('models/model_rf.pkl')
df = pd.read_csv('data/raw/new_traffic.csv')
df['prediction'] = model.predict(df)
df.to_csv('data/processed/predictions.csv', index=False)
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Ensure virtual environment is activated
```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: Dataset download fails
**Solution:** Check internet connection; KDDCup99 downloads from sklearn servers

### Issue: Streamlit not starting
**Solution:** 
```bash
pip install --upgrade streamlit
streamlit run app/app.py --server.port 8502
```

### Issue: Torch import error
**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Performance Expectations

**Dataset:** KDDCup99 (10% sample ≈ 494,000 records)

**Training Times (approximate):**
- Random Forest: 2-5 minutes
- SVM: 10-20 minutes (can be slow on large data)
- KNN: 1-2 minutes
- Isolation Forest: 2-3 minutes
- K-Means: 1-2 minutes
- Autoencoder: 3-5 minutes (CPU), <1 min (GPU)

**Expected Accuracy:** 85-99% depending on model and data quality

---

## Integration with SIEM/SOC

### Option 1: File-Based Integration
```python
# Generate alerts CSV for SIEM ingestion
predictions.to_csv('alerts/ids_alerts.csv')
```

### Option 2: API Endpoint (future enhancement)
```python
# FastAPI wrapper around model
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    return {"prediction": model.predict([data])[0]}
```

### Option 3: ELK Stack Integration
- Export predictions to JSON
- Ingest via Logstash
- Visualize in Kibana

---

## Support & Documentation

- **Full Documentation:** `PROJECT_SUMMARY.md`
- **Technical Report:** `reports/report.tex`
- **Code Examples:** `notebooks/ids_exploration.ipynb`
- **README:** `README.md`

---

## Next Steps After Setup

1. ✅ Train at least one model
2. ✅ Launch the dashboard and explore
3. ✅ Review the LaTeX report
4. ✅ Experiment with custom datasets
5. ✅ Consider production deployment strategies

---

**Last Updated:** January 2026  
**Python:** 3.13+  
**Platform:** Windows 10/11
