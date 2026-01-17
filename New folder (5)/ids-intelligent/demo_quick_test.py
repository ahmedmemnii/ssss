"""Quick demo of IDS system - trains a small RandomForest and shows predictions"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ids.data import load_dataset
from ids.preprocess import split_features_labels, train_val_test_split, make_preprocess_pipeline
from ids.models import build_rf
from ids.evaluate import classification_metrics, print_confusion
from sklearn.pipeline import Pipeline

print("=" * 60)
print("IDS INTELLIGENT - Quick Demo")
print("=" * 60)

# Load dataset
print("\n[1/5] Loading KDDCup99 dataset (10% sample)...")
df = load_dataset(name="kddcup99")
print(f"   ✓ Loaded {len(df)} samples")
print(f"   ✓ Features: {len(df.columns)} columns")
print(f"   ✓ Class distribution:\n{df['binary_label'].value_counts()}")

# Split features/labels
print("\n[2/5] Splitting features and labels...")
X, y = split_features_labels(df, label_col="binary_label")
print(f"   ✓ X shape: {X.shape}, y shape: {y.shape}")

# Train/val/test split
print("\n[3/5] Creating train/validation/test splits...")
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
print(f"   ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Build pipeline
print("\n[4/5] Building preprocessing + RandomForest pipeline...")
preprocess = make_preprocess_pipeline(X_train)
rf = build_rf(n_estimators=50, max_depth=10)  # Small for demo
pipe = Pipeline(steps=[("pre", preprocess), ("clf", rf)])

print("   Training RandomForest...")
pipe.fit(X_train, y_train)
print("   ✓ Training complete!")

# Evaluate
print("\n[5/5] Evaluating on test set...")
y_pred = pipe.predict(X_test)
metrics = classification_metrics(y_test, y_pred, None)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1']:.4f}")

print("\nConfusion Matrix:")
cm = print_confusion(y_test, y_pred)
print(f"              Predicted")
print(f"              Normal  Attack")
print(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"      Attack    {cm[1,0]:5d}   {cm[1,1]:5d}")

print("\n" + "=" * 60)
print("✓ Demo complete! The IDS system is working.")
print("=" * 60)
print("\nNext steps:")
print("1. Train all models: python -m src.ids.train --models rf svm knn iso ae --save")
print("2. Launch dashboard: streamlit run app/app.py")
print("3. Check LaTeX report: reports/report.tex")
