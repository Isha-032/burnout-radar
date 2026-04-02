# BurnoutRadar — AI Early Burnout Detection

> Predict employee burnout **2–3 weeks before it shows** using behavioral signals. Built with XGBoost + SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange) ![Gradio](https://img.shields.io/badge/Gradio-5.23.3-pink) ![License](https://img.shields.io/badge/License-MIT-green)

**Live Demo → [huggingface.co/spaces/Isha-23/burnout-radar](https://huggingface.co/spaces/Isha-23/burnout-radar)**

---

## The Problem

Employee burnout costs companies **$125,000–$190,000 per employee** in turnover, healthcare, and lost productivity *(Gallup, 2023)*. By the time it appears in performance reviews — it's already too late.

BurnoutRadar detects it early by analyzing behavioral signals that shift **weeks before burnout becomes visible**.

---

## Features

- **Team Dashboard** — 12-week burnout trend chart, department heatmap, risk distribution, full employee table
- **Single Employee Analysis** — instant risk score (0–100) with SHAP-powered explanation of top risk drivers
- **Explainable AI** — every prediction shows *why* someone was flagged, not just *that* they were
- **Risk Labels** — Low / Medium / High with actionable HR recommendations per employee
- **Batch CSV Upload** — analyze your entire team at once

---

## How It Works

BurnoutRadar tracks 10 behavioral signals per employee per week:

| Signal | Why It Matters |
|---|---|
| After-hours app usage | #1 predictor — boundary erosion happens first |
| Weekend logins | Inability to disconnect is an early warning sign |
| Task completion rate | Drops weeks before performance reviews flag it |
| Meeting hours / day | Cognitive overload accumulates silently |
| Slack response time | Slows as mental load increases |
| Focus blocks / day | Fragmented attention is a burnout precursor |
| Calendar density | No recovery time leads to inevitable crash |
| Typing speed (WPM) | Decreases under sustained stress |
| Email volume / day | Spikes correlate with overload |
| PTO days used | Low PTO = no recovery |

---

## Model Architecture

```
Raw signals
    ↓
Feature engineering + encoding
    ↓
StandardScaler normalization
    ↓
XGBoost Regressor  →  Burnout Score (0–100)
XGBoost Classifier →  Risk Label (Low / Medium / High)
    ↓
SHAP TreeExplainer →  Top 5 risk drivers per prediction
```

### Performance

| Metric | Value |
|---|---|
| MAE (score regression) | 4.33 points |
| R² | 0.796 |
| Classification accuracy | 88% |
| Training records | 7,500+ |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost 2.1.1 |
| Explainability | SHAP 0.46.0 |
| Data processing | Pandas, NumPy, scikit-learn |
| Visualization | Plotly |
| Frontend / UI | Gradio 5.23.3 |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
burnout-radar/
├── app.py                      # Gradio UI — main application
├── requirements.txt            # Dependencies
├── README.md
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── employee_signals.csv    # Generated dataset (7,500+ records)
└── models/
    ├── train_xgb.py            # Model training script
    ├── regressor.pkl           # Trained XGBoost regressor
    ├── classifier.pkl          # Trained XGBoost classifier
    ├── scaler.pkl              # Feature scaler
    ├── label_encoder.pkl       # Risk label encoder
    ├── dept_encoder.pkl        # Department encoder
    ├── role_encoder.pkl        # Role encoder
    ├── explainer.pkl           # SHAP TreeExplainer
    └── model_meta.json         # Feature names, metrics, classes
```

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/burnout-radar
cd burnout-radar

pip install -r requirements.txt

# Optional: regenerate dataset + retrain models
python data/generate_dataset.py
python models/train_xgb.py

# Launch app
python app.py
```

---

## API Usage

```python
import requests

response = requests.post("https://your-api.com/predict", json={
    "typing_speed_wpm": 55,
    "meeting_hours_per_day": 6,
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
# {
#   "burnout_score": 83.2,
#   "risk_label": "High",
#   "top_factors": [
#     {"feature": "after hours app usage hrs", "impact": 4.2},
#     {"feature": "weekend logins", "impact": 3.1},
#     ...
#   ],
#   "recommendation": "Immediate action needed..."
# }
```

---

## Pricing Model

| Plan | Price | Team Size |
|---|---|---|
| Free | $0 | Up to 10 employees |
| Starter | $99 / month | Up to 100 employees |
| Growth | $299 / month | Up to 500 employees |
| Enterprise | Custom | Unlimited + API access |

---

## Built By

**Isha Patel**
[GitHub](https://github.com/Isha-23) · [Hugging Face](https://huggingface.co/Isha-23)

---

*Built with XGBoost · SHAP · Gradio · Plotly · scikit-learn*

