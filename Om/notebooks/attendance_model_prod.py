"""
attendance_model_prod.py
========================
Production-grade Isolation Forest model for student attendance anomaly detection.
Kalnet AI Project — Om Dattatray (ML Engineer)

Usage:
    python attendance_model_prod.py --mode train   --input data/attendance_features.csv
    python attendance_model_prod.py --mode predict --input data/new_students.csv
    python attendance_model_prod.py --mode evaluate --input data/attendance_features.csv
"""

import os
import sys
import logging
import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("attendance_model.log", encoding="utf-8"),
    ]
)
# Force UTF-8 on Windows stdout to handle special characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
FEATURES = [
    "attendance_rate",
    "longest_absence_streak",
    "absence_in_last_30_days",
    "day_of_week_variance",
]

REQUIRED_COLUMNS = FEATURES + ["student_id"]

MODEL_CONFIG = {
    "n_estimators": 100,
    "contamination": 0.14,   # 14% anomaly rate as defined by Kaif's data synthesis
    "random_state": 42,
    "n_jobs": -1,            # Use all CPU cores
}

PATHS = {
    "model":    "models/attendance_model.pkl",
    "scaler":   "models/attendance_scaler.pkl",
    "metadata": "models/attendance_model_metadata.json",
    "report":   "reports/attendance_evaluation_report.txt",
}

# Thresholds for input validation
VALIDATION_RULES = {
    "attendance_rate":          (0.0, 1.0),
    "longest_absence_streak":   (0, 365),
    "absence_in_last_30_days":  (0, 30),
    "day_of_week_variance":     (0.0, 1.0),
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create required output directories if they don't exist."""
    for path in ["models", "reports"]:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform basic structural validation on the CSV.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def validate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate feature values against expected ranges.
    Returns clean rows and a separate DataFrame of invalid rows.

    Args:
        df: Input DataFrame with feature columns.

    Returns:
        Tuple of (clean_df, invalid_df).
    """
    invalid_mask = pd.Series(False, index=df.index)

    for col, (low, high) in VALIDATION_RULES.items():
        out_of_range = ~df[col].between(low, high)
        if out_of_range.any():
            logger.warning(
                f"Column '{col}': {out_of_range.sum()} values outside [{low}, {high}]"
            )
            invalid_mask |= out_of_range

    invalid_df = df[invalid_mask].copy()
    clean_df   = df[~invalid_mask].copy()

    if len(invalid_df) > 0:
        logger.warning(f"Dropped {len(invalid_df)} invalid rows. See reports/ for details.")
    else:
        logger.info("All rows passed validation.")

    return clean_df, invalid_df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.
    Numeric features -> median imputation.
    Rows where student_id is missing -> dropped.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no NaN values in feature columns.
    """
    if df["student_id"].isna().any():
        before = len(df)
        df = df.dropna(subset=["student_id"])
        logger.warning(f"Dropped {before - len(df)} rows with missing student_id.")

    nan_counts = df[FEATURES].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
        for col in FEATURES:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed '{col}' NaNs with median={median_val:.4f}")

    return df


def save_metadata(model: Pipeline, df: pd.DataFrame, metrics: dict) -> None:
    """Save model metadata and training info to JSON."""
    iso = model.named_steps["isolation_forest"]
    metadata = {
        "model_type":          "IsolationForest",
        "trained_at":          datetime.now().isoformat(),
        "training_samples":    len(df),
        "features":            FEATURES,
        "model_config":        MODEL_CONFIG,
        "n_estimators":        iso.n_estimators,
        "contamination":       iso.contamination,
        "metrics":             metrics,
        "feature_stats": {
            col: {
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()),  4),
                "min":  round(float(df[col].min()),  4),
                "max":  round(float(df[col].max()),  4),
            }
            for col in FEATURES
        }
    }
    with open(PATHS["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved -> {PATHS['metadata']}")


# ── Core Pipeline ──────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline:
      StandardScaler -> IsolationForest

    Scaling is important because absence_streak (0–365) and
    day_of_week_variance (0.0–0.003) are on very different scales.
    Without scaling, high-magnitude features dominate the tree splits.
    """
    return Pipeline([
        ("scaler",          StandardScaler()),
        ("isolation_forest", IsolationForest(**MODEL_CONFIG)),
    ])


def train(filepath: str) -> None:
    """
    Full training pipeline:
      Load -> Validate -> Clean -> Fit -> Save model + metadata.

    Args:
        filepath: Path to labelled training CSV.
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)

    # 1. Load
    df = load_data(filepath)

    # 2. Handle missing values
    df = handle_missing(df)

    # 3. Validate ranges
    df, invalid_df = validate_features(df)
    if len(invalid_df) > 0:
        invalid_df.to_csv("reports/invalid_rows.csv", index=False)

    if len(df) == 0:
        raise ValueError("No valid rows remaining after cleaning. Check your input file.")

    # 4. Extract features
    X = df[FEATURES]

    # 5. Build & train pipeline
    logger.info("Training Isolation Forest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X)
    logger.info("Training complete.")

    # 6. Quick self-evaluation (on training data — for sanity only)
    metrics = {}
    if "is_anomalous" in df.columns:
        y_true = df["is_anomalous"]
        raw    = pipeline.predict(X)
        y_pred = (raw == -1).astype(int)
        report = classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"], output_dict=True)
        metrics = {
            "accuracy":          round(report["accuracy"], 4),
            "anomaly_precision": round(report["Anomalous"]["precision"], 4),
            "anomaly_recall":    round(report["Anomalous"]["recall"], 4),
            "anomaly_f1":        round(report["Anomalous"]["f1-score"], 4),
        }
        logger.info(f"Training accuracy: {metrics['accuracy'] * 100:.1f}%")

    # 7. Save model
    with open(PATHS["model"], "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved -> {PATHS['model']}")

    # 8. Save metadata
    save_metadata(pipeline, df, metrics)

    # 9. Sanity check
    _sanity_check(pipeline)

    logger.info("Training pipeline finished successfully.")


def predict(filepath: str) -> pd.DataFrame:
    """
    Load a trained model and predict on new student data.

    Args:
        filepath: Path to CSV with new students (no is_anomalous column needed).

    Returns:
        DataFrame with predictions appended.
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("PREDICTION MODE")
    logger.info("=" * 60)

    # Load model
    if not Path(PATHS["model"]).exists():
        raise FileNotFoundError(
            f"Model not found at {PATHS['model']}. Run with --mode train first."
        )

    with open(PATHS["model"], "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Model loaded from {PATHS['model']}")

    # Load & clean data
    df = load_data(filepath)
    df = handle_missing(df)
    df, invalid_df = validate_features(df)

    X = df[FEATURES]

    # Predict
    raw    = pipeline.predict(X)
    scores = pipeline.decision_function(X)   # More negative = more anomalous

    df["predicted_anomaly"] = (raw == -1).astype(int)
    df["anomaly_score"]     = np.round(-scores, 4)   # Flip sign: higher = more anomalous
    df["risk_level"]        = pd.cut(
        df["anomaly_score"],
        bins=[-np.inf, 0.05, 0.15, np.inf],
        labels=["Low", "Medium", "High"]
    )

    flagged = df[df["predicted_anomaly"] == 1]
    logger.info(f"Total students processed : {len(df)}")
    logger.info(f"Flagged as anomalous     : {len(flagged)}")

    # Save output
    output_path = "reports/predictions_output.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved -> {output_path}")

    # Print flagged students
    print("\n" + "=" * 60)
    print(f"FLAGGED STUDENTS ({len(flagged)})")
    print("=" * 60)
    print(flagged[["student_id", "attendance_rate",
                   "longest_absence_streak", "absence_in_last_30_days",
                   "anomaly_score", "risk_level"]].to_string(index=False))

    return df


def evaluate(filepath: str) -> None:
    """
    Evaluate the saved model against labelled data.
    Requires 'is_anomalous' column in the CSV.

    Args:
        filepath: Path to labelled CSV.
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)

    if not Path(PATHS["model"]).exists():
        raise FileNotFoundError(
            f"Model not found at {PATHS['model']}. Run with --mode train first."
        )

    with open(PATHS["model"], "rb") as f:
        pipeline = pickle.load(f)

    df = load_data(filepath)
    df = handle_missing(df)
    df, _ = validate_features(df)

    if "is_anomalous" not in df.columns:
        raise ValueError("Evaluation requires 'is_anomalous' column in the CSV.")

    X      = df[FEATURES]
    y_true = df["is_anomalous"]
    raw    = pipeline.predict(X)
    y_pred = (raw == -1).astype(int)

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"])

    # Print to console
    separator = "=" * 60
    output_lines = [
        separator,
        "ATTENDANCE ANOMALY DETECTION — EVALUATION REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset   : {filepath}  ({len(df)} students)",
        separator,
        "",
        "CONFUSION MATRIX",
        "-" * 40,
        f"{'':20s} Pred: Normal   Pred: Anomalous",
        f"{'Actually Normal':20s}    {cm[0][0]:>6}         {cm[0][1]:>6}",
        f"{'Actually Anomalous':20s}    {cm[1][0]:>6}         {cm[1][1]:>6}",
        "",
        "CLASSIFICATION REPORT",
        "-" * 40,
        report,
        "",
        "SANITY CHECK",
        "-" * 40,
    ]

    # Sanity check
    test_cases = pd.DataFrame([
        {"attendance_rate": 0.99, "longest_absence_streak": 1,  "absence_in_last_30_days": 0,  "day_of_week_variance": 0.0001},
        {"attendance_rate": 0.36, "longest_absence_streak": 12, "absence_in_last_30_days": 21, "day_of_week_variance": 0.003},
    ])
    test_raw  = pipeline.predict(test_cases)
    test_pred = (test_raw == -1).astype(int)
    labels    = ["NORMAL" if p == 0 else "ANOMALOUS" for p in test_pred]

    sanity_pass = (labels[0] == "NORMAL") and (labels[1] == "ANOMALOUS")
    output_lines += [
        f"Student with 99% attendance -> {labels[0]:10s}  (expected: NORMAL)",
        f"Student with 36% attendance -> {labels[1]:10s}  (expected: ANOMALOUS)",
        "",
        f"Sanity Check: {'PASSED' if sanity_pass else 'FAILED'}",
        separator,
    ]

    full_output = "\n".join(output_lines)
    print("\n" + full_output)

    # Save report
    with open(PATHS["report"], "w", encoding="utf-8") as f:
        f.write(full_output)
    logger.info(f"Evaluation report saved -> {PATHS['report']}")


# ── Sanity Check (internal) ────────────────────────────────────────────────────

def _sanity_check(pipeline: Pipeline) -> None:
    """Internal sanity check — runs after training."""
    test_cases = pd.DataFrame([
        {"attendance_rate": 0.99, "longest_absence_streak": 1,  "absence_in_last_30_days": 0,  "day_of_week_variance": 0.0001},
        {"attendance_rate": 0.36, "longest_absence_streak": 12, "absence_in_last_30_days": 21, "day_of_week_variance": 0.003},
    ])
    raw   = pipeline.predict(test_cases)
    preds = (raw == -1).astype(int)

    assert preds[0] == 0, "SANITY FAIL: 99% attendance student flagged as anomalous!"
    assert preds[1] == 1, "SANITY FAIL: 36% attendance student NOT flagged!"
    logger.info("Sanity check PASSED.")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalnet AI — Attendance Anomaly Detection (Production)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        required=True,
        help="Operation mode: train | predict | evaluate"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.mode == "train":
            train(args.input)

        elif args.mode == "predict":
            predict(args.input)

        elif args.mode == "evaluate":
            evaluate(args.input)

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data error: {e}")
        sys.exit(1)
    except AssertionError as e:
        logger.critical(f"Sanity check failed: {e}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(99)


if __name__ == "__main__":
    main()
