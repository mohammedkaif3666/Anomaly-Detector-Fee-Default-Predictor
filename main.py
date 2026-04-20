"""
main.py
=======
Entry point for the School Anomaly Detector & Fee Default Predictor.

Runs the full data pipeline:
  1. Generate raw attendance records (500 students x 200 days)
  2. Generate fee records           (500 students x 3 terms)
  3. Engineer attendance features   (4 features + label per student)
  4. Save attendance_features.csv  and  fee_features.csv  to data/
"""

import os
import numpy as np
import pandas as pd

from src.generator import generate_attendance, generate_fees
from src.features  import build_attendance_features


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def verify_attendance(df_raw: pd.DataFrame, df_feat: pd.DataFrame) -> None:
    """Print a quick sanity-check on the generated attendance data."""
    normal_feat    = df_feat[df_feat["is_anomalous"] == 0]
    anomalous_feat = df_feat[df_feat["is_anomalous"] == 1]

    print(f"  Total records  : {len(df_raw):,}  ({df_raw['student_id'].nunique()} students x {df_raw['day'].nunique()} days)")
    print(f"  Normal students: {len(normal_feat)}  | mean attendance rate : {normal_feat['attendance_rate'].mean():.3f}")
    print(f"  Anomalous stus : {len(anomalous_feat)} | mean attendance rate : {anomalous_feat['attendance_rate'].mean():.3f}")
    print(f"  Anomaly rate   : {len(anomalous_feat) / len(df_feat) * 100:.1f}%  (target 14%)")
    print(f"  Avg streak (anomalous) : {anomalous_feat['longest_absence_streak'].mean():.1f} days")
    print(f"  Avg streak (normal)    : {normal_feat['longest_absence_streak'].mean():.1f} days")


def verify_fees(df_fees: pd.DataFrame) -> None:
    """Print distribution check on generated fee data."""
    total = len(df_fees)
    on_time = (df_fees["fee_status"] == "On-time").sum()
    late    = (df_fees["fee_status"] == "Late").sum()
    default = (df_fees["fee_status"] == "Default").sum()

    print(f"  Total rows     : {total}  (500 students x 3 terms)")
    print(f"  On-time        : {on_time:4d}  ({on_time/total*100:.1f}%)  target 80%")
    print(f"  Late           : {late:4d}  ({late/total*100:.1f}%)  target 15%")
    print(f"  Default        : {default:4d}  ({default/total*100:.1f}%)  target  5%")
    print(f"  Columns        : {list(df_fees.columns)}")


# --------------------------------------------------------------------------- #
#  Main pipeline                                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # ---- Step 1: Raw attendance data ----------------------------------------
    section("Step 1 — Generate Attendance Data")
    df_raw_att = generate_attendance(num_students=500, days=200, seed=42)
    print(f"  Raw attendance shape : {df_raw_att.shape}")
    print(f"  Columns : {list(df_raw_att.columns)}")

    # ---- Step 2: Raw fee data -----------------------------------------------
    section("Step 2 — Generate Fee Data")
    df_fees = generate_fees(num_students=500, seed=42)
    verify_fees(df_fees)

    # ---- Step 3: Feature engineering ----------------------------------------
    section("Step 3 — Engineer Attendance Features")
    df_att_features = build_attendance_features(df_raw_att)
    verify_attendance(df_raw_att, df_att_features)
    print(f"\n  Feature columns : {list(df_att_features.columns)}")

    # ---- Step 4: Save to CSV ------------------------------------------------
    section("Step 4 — Save Datasets")

    att_path = os.path.join("data", "attendance_features.csv")
    fee_path = os.path.join("data", "fee_features.csv")

    df_att_features.to_csv(att_path, index=False)
    df_fees.to_csv(fee_path, index=False)

    print(f"  [SAVED] {att_path}  ->  shape {df_att_features.shape}")
    print(f"  [SAVED] {fee_path}   ->  shape {df_fees.shape}")

    # ---- Quick preview ------------------------------------------------------
    section("Preview — attendance_features.csv (first 5 rows)")
    print(df_att_features.head().to_string(index=False))

    section("Preview — fee_features.csv (first 6 rows)")
    print(df_fees.head(6).to_string(index=False))

    print("\n\nPipeline complete. Datasets ready for model training.\n")