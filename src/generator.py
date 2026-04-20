import pandas as pd
import numpy as np

def generate_attendance(num_students=500, days=200):
    """
    Generate attendance data for 500 students over 200 school days.
    - 86% normal: attendance_rate between 80-97%
    - 14% anomalous: sudden drop to 20-40% after a normal start
    """
    data = []
    np.random.seed(42)
    student_ids = [f"STU_{i:03d}" for i in range(num_students)]

    num_anomalous = int(num_students * 0.14)   # 70 students
    anomalous_set = set(student_ids[:num_anomalous])

    for stu_id in student_ids:
        if stu_id not in anomalous_set:
            # Normal: consistent high attendance throughout
            rate = np.random.uniform(0.80, 0.97)
            attendance = np.random.choice([1, 0], size=days, p=[rate, 1 - rate])
            is_anomalous = 0
        else:
            # Anomalous: start normal (days 0-99), SUDDEN DROP (days 100-199)
            normal_rate = np.random.uniform(0.85, 0.97)
            drop_rate   = np.random.uniform(0.20, 0.40)
            first_half  = np.random.choice([1, 0], size=days // 2, p=[normal_rate, 1 - normal_rate])
            second_half = np.random.choice([1, 0], size=days // 2, p=[drop_rate,  1 - drop_rate])
            attendance  = np.concatenate([first_half, second_half])
            is_anomalous = 1

        for day in range(days):
            data.append([stu_id, day, int(attendance[day]), is_anomalous])

    return pd.DataFrame(data, columns=['student_id', 'day', 'present', 'is_anomalous'])


def generate_fees(num_students=500):
    """
    Generate fee data for 500 students across 3 terms.
    - 80% On-time, 15% Late, 5% Default
    - Features: family_income_bracket, transport_user, sibling_count
    - Label: fee_default (1 = Default, 0 = otherwise)
    """
    np.random.seed(42)
    student_ids = [f"STU_{i:03d}" for i in range(num_students)]
    data = []

    for stu_id in student_ids:
        # Student-level features (same across terms)
        family_income_bracket = np.random.choice(['Low', 'Medium', 'High'], p=[0.30, 0.50, 0.20])
        transport_user        = int(np.random.choice([0, 1]))
        sibling_count         = int(np.random.randint(0, 5))

        for term in range(1, 4):   # Term 1, 2, 3
            status = np.random.choice(
                ['On-time', 'Late', 'Default'],
                p=[0.80, 0.15, 0.05]
            )
            fee_default = 1 if status == 'Default' else 0

            data.append([
                stu_id,
                term,
                family_income_bracket,
                transport_user,
                sibling_count,
                status,
                fee_default
            ])

    return pd.DataFrame(data, columns=[
        'student_id',
        'term',
        'family_income_bracket',
        'transport_user',
        'sibling_count',
        'fee_status',
        'fee_default'
    ])