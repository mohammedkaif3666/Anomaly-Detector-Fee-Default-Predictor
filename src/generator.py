"""
generator.py
============
Synthetic data generator for the School Anomaly Detector & Fee Default Predictor.

Attendance Data
---------------
- 500 students  x  200 school days (Mon-Fri cycling)
- 430 normal students  (86%) : attendance_rate ~ Uniform(0.80, 0.97)   → is_anomalous = 0
-  70 anomalous students (14%): start normal, then SUDDEN DROP to       → is_anomalous = 1
   Uniform(0.20, 0.40) at a random breakpoint between day 30 and 150.

Fee Data
--------
- 500 students  x  3 terms
- Overall target distribution : 80% On-time | 15% Late | 5% Default
- Per-student features : family_income_bracket, transport_user, sibling_count
- Income is correlated with default probability for realistic ML signal.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  ATTENDANCE GENERATOR                                                         #
# --------------------------------------------------------------------------- #

def generate_attendance(num_students: int = 500,
                        days: int = 200,
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate raw daily attendance records.

    Parameters
    ----------
    num_students : total students (default 500)
    days         : total school days (default 200)
    seed         : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        student_id | day | day_of_week | present | is_anomalous
    """
    np.random.seed(seed)

    n_normal    = int(num_students * 0.86)          # 430
    n_anomalous = num_students - n_normal           #  70

    # Monte-Carlo rates for all students at once (as in the tip)
    normal_rates    = np.random.uniform(0.80, 0.97, n_normal)
    drop_rates      = np.random.uniform(0.20, 0.40, n_anomalous)
    pre_drop_rates  = np.random.uniform(0.85, 0.97, n_anomalous)  # normal phase
    drop_days       = np.random.randint(30, 151,    n_anomalous)   # random breakpoint

    # Day-of-week: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri  (cycles over 200 days)
    dow_cycle = np.array([d % 5 for d in range(days)])

    records = []

    # --- Normal students ---
    for i in range(n_normal):
        stu_id = f"STU_{i + 1:03d}"
        rate = normal_rates[i]
        present = np.random.binomial(1, rate, days).astype(int)

        for day in range(days):
            records.append((stu_id, day, dow_cycle[day], present[day], 0))

    # --- Anomalous students ---
    for j in range(n_anomalous):
        stu_id    = f"STU_{n_normal + j + 1:03d}"
        pre_rate  = pre_drop_rates[j]
        post_rate = drop_rates[j]
        dp        = drop_days[j]                        # sudden-drop breakpoint

        pre_phase  = np.random.binomial(1, pre_rate,  dp).astype(int)
        post_phase = np.random.binomial(1, post_rate, days - dp).astype(int)
        present    = np.concatenate([pre_phase, post_phase])

        for day in range(days):
            records.append((stu_id, day, dow_cycle[day], present[day], 1))

    df = pd.DataFrame(
        records,
        columns=["student_id", "day", "day_of_week", "present", "is_anomalous"]
    )
    return df


# --------------------------------------------------------------------------- #
#  FEE GENERATOR                                                                #
# --------------------------------------------------------------------------- #

def generate_fees(num_students: int = 500,
                  seed: int = 42) -> pd.DataFrame:
    """
    Generate fee payment records for 3 terms per student.

    Income-bracket correlated default probabilities keep the overall
    distribution close to the target (80% On-time / 15% Late / 5% Default).

    Income bracket mix : 30% Low | 50% Medium | 20% High
    Per-bracket probs  :
        Low    → 10% Default | 22% Late | 68% On-time
        Medium →  3% Default | 13% Late | 84% On-time
        High   →  1% Default |  5% Late | 94% On-time
    Weighted average   → ~5% Default | ~14% Late | ~81% On-time  ✓

    Parameters
    ----------
    num_students : total students (default 500)
    seed         : random seed

    Returns
    -------
    pd.DataFrame with columns:
        student_id | term | family_income_bracket | transport_user |
        sibling_count | fee_status | fee_default
    """
    np.random.seed(seed)

    # Per-bracket payment probability vectors [On-time, Late, Default]
    bracket_probs = {
        "Low":    [0.68, 0.22, 0.10],
        "Medium": [0.84, 0.13, 0.03],
        "High":   [0.94, 0.05, 0.01],
    }
    statuses = ["On-time", "Late", "Default"]

    records = []

    for i in range(num_students):
        stu_id = f"STU_{i + 1:03d}"

        # ---- Student-level demographic features (fixed across terms) --------
        family_income_bracket = np.random.choice(
            ["Low", "Medium", "High"], p=[0.30, 0.50, 0.20]
        )
        transport_user = int(np.random.choice([0, 1], p=[0.55, 0.45]))
        sibling_count  = int(np.random.randint(0, 5))   # 0 to 4

        probs = bracket_probs[family_income_bracket]

        # ---- 3 terms per student --------------------------------------------
        for term in range(1, 4):
            fee_status  = np.random.choice(statuses, p=probs)
            fee_default = 1 if fee_status == "Default" else 0

            records.append((
                stu_id, term,
                family_income_bracket, transport_user, sibling_count,
                fee_status, fee_default
            ))

    df = pd.DataFrame(
        records,
        columns=[
            "student_id", "term",
            "family_income_bracket", "transport_user", "sibling_count",
            "fee_status", "fee_default"
        ]
    )
    return df