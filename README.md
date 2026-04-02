---
title: BurnoutRadar — AI Early Burnout Detector
emoji: 🔥
colorFrom: teal
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Predict employee burnout 2–3 weeks before it shows
---

# 🔥 BurnoutRadar — AI Early Burnout Detector

**Predict employee burnout 2–3 weeks before it becomes visible** using behavioral signals like typing speed, meeting load, after-hours usage, and calendar density.

## Features

- **Single employee** — enter signals manually, get instant risk score + SHAP explanation
- **Batch CSV** — upload your whole team, get a risk dashboard
- **Explainable AI** — every prediction shows top 3 risk drivers (no black box)
- **REST API** — integrate directly into your HRIS

## How it works

1. Collects 10 behavioral signals per employee per week
2. XGBoost Regressor outputs a burnout score (0–100)
3. XGBoost Classifier labels risk: Low / Medium / High
4. SHAP values explain which signals drove the score
5. HR gets actionable recommendations

## Model performance

| Metric | Value |
|---|---|
| MAE (score) | 4.33 points |
| R² | 0.796 |
| Accuracy (label) | 88% |

## Signals tracked

| Signal | Importance |
|---|---|
| After-hours app usage | ⭐⭐⭐⭐⭐ |
| Weekend logins | ⭐⭐⭐⭐ |
| Task completion rate | ⭐⭐⭐⭐ |
| Meeting hours/day | ⭐⭐⭐ |
| Slack response time | ⭐⭐⭐ |

## Tech stack

XGBoost · SHAP · Gradio · FastAPI · Plotly · scikit-learn

## API

```python
import requests

response = requests.post("https://your-api.com/predict", json={
    "typing_speed_wpm": 55,
    "meeting_hours_per_day": 6.5,
    "after_hours_app_usage_hrs": 3.5,
    "weekend_logins": 8,
    "calendar_density": 0.8,
    "slack_response_time_min": 45,
    "task_completion_rate": 0.62,
    "pto_days_used": 0,
    "focus_time_blocks": 1.0,
    "email_volume_per_day": 60,
    "department": "Engineering",
    "role": "Senior"
})
print(response.json())
```

## License

MIT — free to use, modify, and deploy commercially.

---

Built by [your name] · [LinkedIn] · [Twitter]
