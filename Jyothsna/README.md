# Jyothsna — Kalnet AI Predictor API

This API wraps the Fee Default Predictor and Attendance Anomaly Detection models.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API (from the project root):
   ```bash
   python Jyothsna/main.py
   ```
   Or using uvicorn directly from the project root:
   ```bash
   uvicorn Jyothsna.main:app --reload
   ```

## Endpoints

### 1. Fee Default Prediction
- **URL**: `/predict/fee-default`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "student_id": "STU001",
    "current_term": 1,
    "income_encoded": 1,
    "transport_user": 1,
    "sibling_count": 2,
    "current_status": 0,
    "current_defaulted": 0,
    "is_low_income": 0,
    "has_many_siblings": 0,
    "was_late_or_worse": 0
  }
  ```

### 2. Attendance Anomaly Detection
- **URL**: `/predict/attendance-anomaly`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "student_id": "STU001",
    "attendance_rate": 0.95,
    "longest_absence_streak": 2,
    "absence_in_last_30_days": 1,
    "day_of_week_variance": 0.001
  }
  ```

## Documentation
Once the API is running, you can access the interactive Swagger documentation at:
- http://localhost:8000/docs
