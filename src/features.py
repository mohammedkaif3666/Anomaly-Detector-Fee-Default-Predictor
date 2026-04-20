import pandas as pd
import numpy as np


def build_attendance_features(df_att):
    """
    Build per-student attendance feature matrix from raw daily attendance data.

    Features:
    - attendance_rate         : overall % of days present
    - longest_absence_streak  : max consecutive days absent
    - absence_in_last_30_days : # of absences in the final 30 days
    - day_of_week_variance    : variance of attendance across simulated weekly slots
    - label                   : 1 = anomalous, 0 = normal
    """
    features = []

    for stu_id, group in df_att.groupby('student_id'):
        # Sort by day to ensure chronological order
        group = group.sort_values('day')
        attendance_array = group['present'].values  # array of 0s and 1s

        # 1. Attendance Rate
        attendance_rate = attendance_array.mean()

        # 2. Longest Absence Streak
        #    Convert to string of 0/1, split on '1' (present) to get absence runs
        absence_str = "".join(map(str, 1 - attendance_array))
        longest_absence_streak = max((len(s) for s in absence_str.split('1')), default=0)

        # 3. Absence in Last 30 Days
        absence_in_last_30_days = int(30 - attendance_array[-30:].sum())

        # 4. Day-of-Week Variance
        #    Simulated: split 200 days into 40 weeks of 5 days each,
        #    compute mean attendance per week, then variance across weeks
        weekly_attendance = attendance_array.reshape(-1, 5).mean(axis=1)
        day_of_week_variance = float(np.var(weekly_attendance))

        # Label
        label = int(group['is_anomalous'].iloc[0])

        features.append([
            stu_id,
            round(attendance_rate, 4),
            longest_absence_streak,
            absence_in_last_30_days,
            round(day_of_week_variance, 6),
            label
        ])

    return pd.DataFrame(features, columns=[
        'student_id',
        'attendance_rate',
        'longest_absence_streak',
        'absence_in_last_30_days',
        'day_of_week_variance',
        'label'
    ])