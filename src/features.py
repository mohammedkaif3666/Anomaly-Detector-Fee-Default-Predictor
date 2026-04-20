import pandas as pd
import numpy as np

def build_features(df_att):
    features = []
    
    for stu_id, group in df_att.groupby('student_id'):
        attendance_array = group['present'].values
        
        # 1. Attendance Rate
        rate = attendance_array.mean()
        
        # 2. Longest Absence Streak
        absences = "".join(map(str, 1 - attendance_array))
        streak = max([len(s) for s in absences.split('0')] + [0])
        
        # 3. Absence in last 30 days
        last_30 = 30 - attendance_array[-30:].sum()
        
        # 4. Day of week variance (simulated by splitting 200 days into 5-day chunks)
        # We'll calculate the variance of attendance across the 40 weeks
        weekly_att = attendance_array.reshape(-1, 5).mean(axis=1)
        variance = np.var(weekly_att)
        
        label = group['is_anomalous'].iloc[0]
        
        features.append([stu_id, rate, streak, last_30, variance, label])
        
    return pd.DataFrame(features, columns=['student_id', 'attendance_rate', 'streak', 'last_30_absent', 'dow_variance', 'label'])