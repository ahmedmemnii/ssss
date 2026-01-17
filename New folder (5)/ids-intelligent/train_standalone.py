import sys
import os

# Add project to path
sys.path.insert(0, r"C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent")

print("=" * 70)
print("TRAINING IDS MODEL - STANDALONE")
print("=" * 70)

try:
    print("\n[1/6] Importing libraries...")
    from src.ids.data import load_dataset
    from src.ids.preprocess import split_features_labels, train_val_test_split, make_preprocess_pipeline
    from src.ids.models import build_rf
    from src.ids.evaluate import classification_metrics, print_confusion
    from sklearn.pipeline import Pipeline
    import joblib
    from pathlib import Path
    print("✓ Imports successful")

    print("\n[2/6] Loading KDDCup99 dataset...")
    df = load_dataset(name="kddcup99")
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")

    print("\n[3/6] Preprocessing data...")
    X, y = split_features_labels(df, label_col="binary_label")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n[4/6] Building and training Random Forest...")
    preprocess = make_preprocess_pipeline(X_train)
    rf = build_rf()  # Use defaults
    pipe = Pipeline(steps=[("pre", preprocess), ("clf", rf)])
    pipe.fit(X_train, y_train)
    print("✓ Training complete")

    print("\n[5/6] Evaluating on test set...")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
    metrics = classification_metrics(y_test, y_pred, y_proba)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    cm = print_confusion(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Attack")
    print(f"Actual Normal   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"      Attack    {cm[1,0]:5d}   {cm[1,1]:5d}")

    print("\n[6/6] Saving model...")
    models_dir = Path(r"C:\Users\Have Fun\Desktop\New folder (5)\ids-intelligent\models")
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / "model_rf.pkl"
    joblib.dump(pipe, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save performance
    import pandas as pd
    perf_df = pd.DataFrame([{"model": "rf", **metrics}])
    perf_path = models_dir / "performance.csv"
    perf_df.to_csv(perf_path, index=False)
    print(f"✓ Performance saved to: {perf_path}")

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext: Run 'streamlit run app/app.py' to launch dashboard")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
