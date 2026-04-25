# Anomaly Detector & Fee Default Predictor

> Machine Learning in Real Schools — Scikit-learn only, no external API needed.

---

## The Vision

The admin opens KALNET on Friday.  
A **red section** appears: *3 students flagged as high attendance risk this week.*  
She clicks. **Rahul Sharma, Class 9A** — attendance dropped from 92% to 34% in 3 weeks. Flagged as anomalous.  
The admin calls his parents. A family issue is discovered.  
Without this system, no one would have noticed for another month.

The finance section shows **5 students predicted to miss next term's payment**.  
The team reaches out early. One default is prevented.

**This is machine learning in real schools.**

---

## Project Structure

```text
school_predictor/
├── data/                         # Generated synthetic datasets go here
│   ├── attendance_features.csv   # 1 row/student, 4 features + label
│   └── fee_features.csv          # 3 rows/student (terms), 5 col + label
├── src/                          # Data Generation Source Code
│   ├── generator.py              # Synthetic data generation (NumPy)
│   └── features.py               # Feature engineering
├── Om/                           # Attendance Anomaly Detection Model (ML)
│   ├── notebooks/
│   │   ├── attendance_model_prod.py    # Production pipeline for isolation forest
│   │   └── train_attendance_model.py   # Simple training script
│   ├── models/                   # Saved Isolation Forest model (.pkl)
│   └── reports/                  # Evaluation reports and CSV outputs
├── Are_Samhith/                  # Fee Default Prediction Model (ML)
│   └── models/
│       ├── train_v2_final.py     # GradientBoostingClassifier for fee default
│       ├── model_v2.pkl          # Saved GradientBoosting model
│       └── threshold_v2.pkl      # Tuned decision threshold (0.40)
├── main.py                       # Data generation pipeline runner
├── requirements.txt              # Project dependencies
└── README.md
```

---

## My Contribution — Dataset Generation

### Attendance Data (`attendance_features.csv`)

| Spec | Value |
|---|---|
| Students | 500 |
| School days | 200 (Mon–Fri cycling) |
| Normal students (86%) | `attendance_rate ~ Uniform(0.80, 0.97)` → `is_anomalous = 0` |
| Anomalous students (14%) | Start normal, then sudden drop. → `is_anomalous = 1` |

**Features:** `attendance_rate`, `longest_absence_streak`, `absence_in_last_30_days`, `day_of_week_variance`.

### Fee Data (`fee_features.csv`)

| Spec | Value |
|---|---|
| Students | 500 |
| Terms | 3 per student (1500 rows total) |
| Default | ~5% |

**Features:** `family_income_bracket`, `transport_user`, `sibling_count`, `fee_status` (Term N features predict Term N+1 default).

---

## ML Models

### 1. Attendance Anomaly Detection (by Om)
Uses an **Isolation Forest** to identify students with highly irregular attendance patterns (e.g., a sudden drop from 95% to 30%).
- **How to train:** `cd Om && python notebooks/attendance_model_prod.py --mode train --input ../data/attendance_features.csv`
- **Output:** Saves `attendance_model.pkl` in `Om/models/`.

### 2. Fee Default Predictor (by Are Samhith)
Uses a **GradientBoostingClassifier** with `sample_weight="balanced"` and a tuned threshold (0.40) to hit a 70%+ recall rate in predicting next-term fee defaults.
- **How to train:** `cd Are_Samhith/models && python train_v2_final.py`
- **Output:** Saves `model_v2.pkl` and `threshold_v2.pkl`.

---

## How to Run the Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate datasets (creates files in data/)
python main.py

# 3. Train models
cd Om && python notebooks/train_attendance_model.py
cd ../Are_Samhith/models && python train_v2_final.py
```

---

## Design Decisions
- **Synthetic data only** — generated with NumPy. No real dataset downloaded.
- **Sudden-drop anomaly** — anomalous students look normal at first, then attendance collapses.
- **Correlated features** — income bracket influences default probability, making the fee model learn meaningful patterns.
- **No data leakage** — Fee default model strictly uses Term N to predict Term N+1.
