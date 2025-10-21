# Beginner-friendly training script (updated for numeric-feature datasets too)
# Usage:
#   python train_model.py --csv dataset.csv --label-col CLASS_LABEL --out phishing_model.pkl

import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from feature_extraction import build_feature_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with URL or features + label")
    ap.add_argument("--url-col", default=None, help="Column name for URL (only if dataset has raw URLs)")
    ap.add_argument("--label-col", default="Label", help="Column name for label (1=phish,0=legit)")
    ap.add_argument("--out", default="phishing_model.pkl", help="Output model filename")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # --------------------------
    # Case 1: Dataset has raw URLs
    # --------------------------
    if args.url_col and args.url_col in df.columns:
        print(f"Using URL column '{args.url_col}' with feature extraction...")
        X = build_feature_frame(df[args.url_col])
        y = df[args.label_col].astype(int)

    # --------------------------
    # Case 2: Dataset already has numeric features
    # --------------------------
    else:
        print("No raw URL column found → using all numeric features directly")
        # Drop non-numeric columns (like 'id')
        feature_cols = [c for c in df.columns if c != args.label_col]
        X = df[feature_cols].select_dtypes(include=['int64','float64'])
        y = df[args.label_col].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump({"model": model, "features": list(X.columns)}, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
