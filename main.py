from src.generator import generate_attendance, generate_fees
from src.features import build_features
import os

# Create data folder if not exists
if not os.path.exists('data'): os.makedirs('data')

print("Generating data...")
df_raw_att = generate_attendance()
df_fees = generate_fees()

print("Engineering features...")
df_att_features = build_features(df_raw_att)

# Save files
df_att_features.to_csv('data/attendance_features.csv', index=False)
df_fees.to_csv('data/fee_features.csv', index=False)

print("Done! Check the 'data' folder.")