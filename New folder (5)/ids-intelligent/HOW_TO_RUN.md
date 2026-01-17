# 🎯 HOW TO RUN THE IDS PROJECT

## ⚠️ IMPORTANT: You Were in the Wrong Directory!

The error occurred because you were in:
```
C:\Users\Have Fun\Desktop\New folder (5)\
```

But the project is in:
```
C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent\
```

---

## ✅ CORRECT WAY TO RUN (3 Options)

### **Option 1: Use the Interactive Menu (EASIEST)**

1. Double-click: `START.bat`
2. Choose from the menu:
   - `[1]` Train Random Forest only (5 min)
   - `[2]` Train all models (20-30 min)
   - `[3]` Launch dashboard
   - `[4]` Verify setup

---

### **Option 2: Use PowerShell Commands (Manual)**

```powershell
# STEP 1: Navigate to the project directory
cd "C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent"

# STEP 2: Train a model (choose one)
# Quick (RF only - 5 min):
& "C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\python.exe" -m src.ids.train --dataset kddcup99 --models rf --save

# OR Full (all models - 20-30 min):
& "C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\python.exe" -m src.ids.train --dataset kddcup99 --models rf svm knn iso kmeans ae --save

# STEP 3: Launch dashboard
& "C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\streamlit.exe" run app\app.py
```

---

### **Option 3: Activate Virtual Environment First**

```powershell
# Navigate to project
cd "C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent"

# Activate venv
& "C:\Users\Have Fun\Desktop\New folder (5)\.venv\Scripts\Activate.ps1"

# Now you can use simpler commands
python -m src.ids.train --dataset kddcup99 --models rf --save
streamlit run app\app.py
```

---

## 📊 WHAT EACH COMMAND DOES

### Training Command
```bash
python -m src.ids.train --dataset kddcup99 --models rf svm knn iso kmeans ae --save
```

**What happens:**
1. Downloads KDDCup99 dataset (~50MB, first time only)
2. Trains each model:
   - `rf` = Random Forest
   - `svm` = Support Vector Machine
   - `knn` = K-Nearest Neighbors
   - `iso` = Isolation Forest
   - `kmeans` = K-Means clustering
   - `ae` = Autoencoder (Deep Learning)
3. Saves models to `models/` folder
4. Creates `models/performance.csv` with comparison results
5. Prints confusion matrices and metrics

**Output Example:**
```
Training model: rf
Accuracy: 0.9876
Precision: 0.9801
Recall: 0.9912
F1: 0.9856

Confusion Matrix:
              Predicted
              Normal  Attack
Actual Normal  95234    1234
      Attack     456   98765

Saved performance to models/performance.csv
```

---

### Dashboard Command
```bash
streamlit run app\app.py
```

**What happens:**
1. Starts web server on http://localhost:8501
2. Opens browser automatically
3. Shows dashboard with:
   - Model selector dropdown
   - CSV upload button
   - Prediction results
   - Anomaly detection chart
   - Summary metrics

---

## 🔧 TROUBLESHOOTING

### Error: "No module named 'src'"
**Cause:** Wrong directory  
**Fix:** Make sure you're in `ids-intelligent` folder:
```powershell
cd "C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent"
```

### Error: "File does not exist: app/app.py"
**Cause:** Wrong directory  
**Fix:** Same as above

### Error: "pdflatex not found"
**Cause:** LaTeX not installed  
**Fix:** The `.tex` file IS the deliverable. You can:
1. Upload it to Overleaf.com to compile online
2. Install MiKTeX or TeX Live if you need the PDF
3. Submit the `.tex` file as-is

---

## 📁 PROJECT FILES LOCATION

```
C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent\
├── START.bat                  ← Double-click this for menu
├── RUN_PROJECT.bat           ← Auto-train script
├── demo_quick_test.py        ← Python demo
├── verify_setup.bat          ← Check installation
│
├── src/ids/                  ← Core Python modules
│   ├── data.py
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   └── app.py               ← Streamlit dashboard
│
├── reports/
│   └── report.tex           ← LaTeX report (deliverable)
│
├── notebooks/
│   └── ids_exploration.ipynb
│
└── models/                  ← Models saved here after training
    ├── model_rf.pkl
    ├── model_svm.pkl
    └── performance.csv
```

---

## 🚀 RECOMMENDED WORKFLOW

### For Quick Demo (10 minutes):
1. Run `START.bat`
2. Choose option `[1]` (train RF only)
3. Wait 5 minutes for training
4. Choose option `[3]` to launch dashboard
5. Upload a test CSV or view results

### For Complete Project (30 minutes):
1. Run `START.bat`
2. Choose option `[2]` (train all models)
3. Wait 20-30 minutes
4. View `models/performance.csv` for comparison
5. Launch dashboard to test predictions

### To Just View the Report:
1. Open `reports/report.tex` in any text editor
2. Or upload to Overleaf.com to compile to PDF
3. Or submit the .tex file directly

---

## ✅ PROJECT STATUS

All components are complete and functional:
- ✅ 6 ML/DL models implemented
- ✅ Training pipeline working
- ✅ Evaluation metrics implemented
- ✅ Streamlit dashboard ready
- ✅ LaTeX report written
- ✅ Documentation complete

**The project is 100% ready for demonstration!**

---

## 💡 TIPS

1. **First Time:** Run option `[1]` first to test quickly
2. **Full Demo:** Run option `[2]` for complete model comparison
3. **Dashboard:** Needs trained models to work (run training first)
4. **Report:** The .tex file is the deliverable, PDF compilation is optional

---

## 📞 QUICK HELP

**Problem:** Nothing works  
**Solution:** Double-click `verify_setup.bat` to check installation

**Problem:** Training is slow  
**Solution:** Normal! SVM takes 10-20 minutes. Use option `[1]` for faster demo.

**Problem:** Want to see the report  
**Solution:** Open `reports\report.tex` in Notepad or upload to Overleaf.com

---

**Last Updated:** January 17, 2026  
**Status:** Ready for demonstration
