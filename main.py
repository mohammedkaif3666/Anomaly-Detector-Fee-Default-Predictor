from src.generator import generate_attendance, generate_fees
from src.features import build_attendance_features
import os

# Create data folder if not exists
if not os.path.exists('data'):
    os.makedirs('data')

print("=" * 50)
print("  Anomaly Detector & Fee Default Predictor")
print("=" * 50)

# --- Step 1: Generate raw data ---
print("\n[1/3] Generating attendance data (500 students x 200 days)...")
df_raw_att = generate_attendance(num_students=500, days=200)
print(f"      Rows generated: {len(df_raw_att):,}  |  Anomalous students: {df_raw_att[df_raw_att['is_anomalous']==1]['student_id'].nunique()}")

print("\n[2/3] Generating fee data (500 students x 3 terms)...")
df_fees = generate_fees(num_students=500)
print(f"      Rows generated: {len(df_fees):,}")
print(f"      On-time: {(df_fees['fee_status']=='On-time').sum()} | Late: {(df_fees['fee_status']=='Late').sum()} | Default: {(df_fees['fee_status']=='Default').sum()}")

# --- Step 2: Feature engineering ---
print("\n[3/3] Engineering attendance features...")
df_att_features = build_attendance_features(df_raw_att)
print(f"      Features built for {len(df_att_features)} students")
print(f"      Columns: {list(df_att_features.columns)}")

# --- Step 3: Save to CSV ---
att_path = 'data/attendance_features.csv'
fee_path = 'data/fee_features.csv'

df_att_features.to_csv(att_path, index=False)
df_fees.to_csv(fee_path, index=False)

print(f"\n[SAVED] {att_path}")
print(f"[SAVED] {fee_path}")
print("\nDone! Check the 'data/' folder.")
print("=" * 50)