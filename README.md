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

```
school_predictor/
├── data/
│   ├── attendance_features.csv   # one row per student, 4 features + label
│   └── fee_features.csv          # 3 rows per student (terms), 5 col + label
├── src/
│   ├── generator.py              # synthetic data generation (NumPy)
│   └── features.py               # feature engineering
├── main.py                       # full pipeline runner
├── requirements.txt
└── README.md
```

---

## My Contribution — Dataset Generation

### Attendance Data  (`attendance_features.csv`)

| Spec | Value |
|---|---|
| Students | 500 |
| School days | 200 (Mon–Fri cycling) |
| Normal students (86%) | `attendance_rate ~ Uniform(0.80, 0.97)` → `is_anomalous = 0` |
| Anomalous students (14%) | Start normal (85-97%), then **sudden drop** to `Uniform(0.20, 0.40)` at a random breakpoint (day 30–150) → `is_anomalous = 1` |

**Engineered features per student:**

| Feature | Description |
|---|---|
| `attendance_rate` | Fraction of 200 days the student was present |
| `longest_absence_streak` | Maximum consecutive days absent |
| `absence_in_last_30_days` | Absences in the final 30 school days |
| `day_of_week_variance` | Variance of per-weekday attendance rates (Mon–Fri) |
| `is_anomalous` | **Label** — 1 = anomalous, 0 = normal |

### Fee Data  (`fee_features.csv`)

| Spec | Value |
|---|---|
| Students | 500 |
| Terms | 3 per student (1500 rows total) |
| On-time | ~80% |
| Late | ~15% |
| Default | ~5% |

**Features:**

| Feature | Description |
|---|---|
| `family_income_bracket` | Low / Medium / High (30% / 50% / 20%) |
| `transport_user` | 1 = uses school transport, 0 = does not |
| `sibling_count` | Number of siblings (0–4) |
| `fee_status` | On-time / Late / Default |
| `fee_default` | **Label** — 1 = Default, 0 = On-time or Late |

> Income is correlated with default probability — Low-income families have higher default risk — providing a realistic ML signal.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets
python main.py
```

Datasets will be saved to `data/`.

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

---

## Design Decisions

- **Synthetic data only** — generated with NumPy. No real dataset downloaded.
- **Sudden-drop anomaly** — anomalous students look normal at first (days 0–breakpoint), then attendance collapses. This is what makes the problem hard and realistic.
- **Random breakpoints** — each anomalous student's drop happens at a different day (30–150), preventing the model from learning a trivial time-position shortcut.
- **Correlated features** — income bracket influences default probability, making the fee model actually learn meaningful patterns.
- **Reproducibility** — all generators use `seed=42`.
