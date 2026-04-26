from pydantic import BaseModel
from typing import List, Optional

class FeeDefaultRequest(BaseModel):
    student_id: str
    current_term: int
    income_encoded: int
    transport_user: int
    sibling_count: int
    current_status: int
    current_defaulted: int
    is_low_income: int
    has_many_siblings: int
    was_late_or_worse: int

class FeeDefaultResponse(BaseModel):
    student_id: str
    default_probability: float
    risk_category: str
    will_default_pred: int

class AttendanceAnomalyRequest(BaseModel):
    student_id: str
    attendance_rate: float
    longest_absence_streak: int
    absence_in_last_30_days: int
    day_of_week_variance: float

class AttendanceAnomalyResponse(BaseModel):
    student_id: str
    predicted_anomaly: int
    anomaly_score: float
    risk_level: str
