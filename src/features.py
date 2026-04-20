"""
features.py
===========
Feature engineering for attendance anomaly detection.

Takes the raw daily attendance DataFrame (output of generate_attendance)
and collapses it into one summary row per student with four key features:

    attendance_rate          – fraction of 200 days the student was present
    longest_absence_streak   – maximum consecutive days absent
    absence_in_last_30_days  – number of absences in the final 30 school days
    day_of_week_variance     – variance of per-weekday attendance rates
                               (Mon / Tue / Wed / Thu / Fri mean rates → var)
                               High variance → student consistently skips
                               specific days, a meaningful anomaly signal.

Label column:
    is_anomalous             – 1 = anomalous student, 0 = normal
"""

import numpy as np
import pandas as pd


def build_attendance_features(df_att: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer per-student feature matrix from raw daily attendance records.

    Parameters
    ----------
    df_att : pd.DataFrame
        Output of generate_attendance(). Expected columns:
        student_id, day, day_of_week, present, is_anomalous

    Returns
    -------
    pd.DataFrame with one row per student and columns:
        student_id | attendance_rate | longest_absence_streak |
        absence_in_last_30_days | day_of_week_variance | is_anomalous
    """
    records = []

    for stu_id, grp in df_att.groupby("student_id"):
        # Ensure chronological order
        grp = grp.sort_values("day").reset_index(drop=True)
        present = grp["present"].values      # shape: (200,), values: 0 or 1

        # ------------------------------------------------------------------ #
        # 1.  Attendance Rate                                                  #
        #     Simple mean over all 200 days.                                   #
        # ------------------------------------------------------------------ #
        attendance_rate = round(float(present.mean()), 4)

        # ------------------------------------------------------------------ #
        # 2.  Longest Absence Streak                                           #
        #     Walk the array and track consecutive zeros.                      #
        # ------------------------------------------------------------------ #
        longest_absence_streak = 0
        current_run = 0
        for p in present:
            if p == 0:
                current_run += 1
                if current_run > longest_absence_streak:
                    longest_absence_streak = current_run
            else:
                current_run = 0

        # ------------------------------------------------------------------ #
        # 3.  Absence in Last 30 Days                                          #
        #     Number of absent days in the most recent 30 school days.         #
        # ------------------------------------------------------------------ #
        absence_in_last_30_days = int((1 - present[-30:]).sum())

        # ------------------------------------------------------------------ #
        # 4.  Day-of-Week Variance                                             #
        #     For each weekday (Mon=0 … Fri=4) compute the student's           #
        #     attendance rate on that specific day, then take the variance     #
        #     across those 5 rates.                                            #
        #     A normal student → low variance (similar on every day).          #
        #     An anomalous student → high variance (misses blocks of days).    #
        # ------------------------------------------------------------------ #
        dow_rates = []
        for dow in range(5):          # 0=Mon … 4=Fri
            dow_mask = grp["day_of_week"] == dow
            rate_on_day = grp.loc[dow_mask, "present"].mean()
            dow_rates.append(rate_on_day)
        day_of_week_variance = round(float(np.var(dow_rates)), 6)

        # ------------------------------------------------------------------ #
        # Label                                                                #
        # ------------------------------------------------------------------ #
        is_anomalous = int(grp["is_anomalous"].iloc[0])

        records.append({
            "student_id"             : stu_id,
            "attendance_rate"        : attendance_rate,
            "longest_absence_streak" : longest_absence_streak,
            "absence_in_last_30_days": absence_in_last_30_days,
            "day_of_week_variance"   : day_of_week_variance,
            "is_anomalous"           : is_anomalous,
        })

    return pd.DataFrame(records)