import pandas as pd
import numpy as np

def generate_attendance(num_students=500, days=200):
    data = []
    student_ids = [f"STU_{i:03d}" for i in range(num_students)]
    
    # Split: 86% normal, 14% anomalous
    num_anomalous = int(num_students * 0.14)
    anomalous_ids = student_ids[:num_anomalous]
    normal_ids = student_ids[num_anomalous:]

    for stu_id in student_ids:
        if stu_id in normal_ids:
            # Random rate between 80-97%
            rate = np.random.uniform(0.80, 0.97)
            attendance = np.random.choice([1, 0], size=days, p=[rate, 1-rate])
        else:
            # Anomalous: Drop to 20-40%
            rate = np.random.uniform(0.20, 0.40)
            attendance = np.random.choice([1, 0], size=days, p=[rate, 1-rate])
        
        for day in range(days):
            data.append([stu_id, day, attendance[day], 1 if stu_id in anomalous_ids else 0])
            
    return pd.DataFrame(data, columns=['student_id', 'day', 'present', 'is_anomalous'])

def generate_fees(num_students=500):
    student_ids = [f"STU_{i:03d}" for i in range(num_students)]
    data = []
    
    for stu_id in student_ids:
        # Features
        income = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
        transport = np.random.choice([0, 1])
        siblings = np.random.randint(0, 4)
        
        # Payment Status (80% On-time, 15% Late, 5% Default)
        status = np.random.choice(['On-time', 'Late', 'Default'], p=[0.80, 0.15, 0.05])
        
        data.append([stu_id, income, transport, siblings, status])
        
    return pd.DataFrame(data, columns=['student_id', 'income_bracket', 'transport_user', 'sibling_count', 'fee_status'])