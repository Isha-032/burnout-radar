import gradio as gr
import pandas as pd
import numpy as np
import pickle, json, shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Load models ───────────────────────────────────────────────
with open("models/regressor.pkl","rb") as f: reg = pickle.load(f)
with open("models/classifier.pkl","rb") as f: clf = pickle.load(f)
with open("models/scaler.pkl","rb") as f: scaler = pickle.load(f)
with open("models/label_encoder.pkl","rb") as f: le_label = pickle.load(f)
with open("models/dept_encoder.pkl","rb") as f: le_dept = pickle.load(f)
with open("models/role_encoder.pkl","rb") as f: le_role = pickle.load(f)
with open("models/explainer.pkl","rb") as f: explainer = pickle.load(f)
with open("models/model_meta.json") as f: meta = json.load(f)

FEATURES = meta["features"]
DEPTS    = meta["departments"]
ROLES    = meta["roles"]
RC = {"High":"#E24B4A","Medium":"#EF9F27","Low":"#1D9E75"}
RB = {"High":"#FCEBEB","Medium":"#FAEEDA","Low":"#E1F5EE"}

# ── Realistic demo employees ──────────────────────────────────
DEMO_EMPLOYEES = [
    {"name":"Arjun Mehta",    "dept":"Engineering","role":"Senior",  "trend":[22,24,28,31,38,44,52,61,68,74,79,83]},
    {"name":"Priya Sharma",   "dept":"Marketing",  "role":"Lead",    "trend":[18,19,21,20,22,24,23,25,27,26,28,30]},
    {"name":"Rohan Desai",    "dept":"Sales",      "role":"Mid",     "trend":[35,38,42,47,53,59,63,67,72,76,80,85]},
    {"name":"Sneha Patel",    "dept":"HR",         "role":"Manager", "trend":[15,16,14,17,15,18,16,19,17,20,18,21]},
    {"name":"Karan Joshi",    "dept":"Engineering","role":"Junior",  "trend":[28,31,35,40,46,50,55,60,64,69,73,77]},
    {"name":"Meera Nair",     "dept":"Finance",    "role":"Senior",  "trend":[20,22,21,23,22,24,25,24,26,25,27,28]},
    {"name":"Dev Agarwal",    "dept":"Operations", "role":"Lead",    "trend":[40,44,49,54,60,65,70,74,78,81,84,87]},
    {"name":"Ananya Singh",   "dept":"Engineering","role":"Mid",     "trend":[25,27,29,28,30,32,31,33,35,34,36,38]},
]

def make_signals(score_pct):
    """Generate realistic signals from a burnout score (0-100)"""
    s = score_pct / 100
    return {
        "typing_speed_wpm":          max(20, np.random.normal(75 - s*30, 3)),
        "meeting_hours_per_day":     np.clip(np.random.normal(2 + s*5, 0.5), 0, 10),
        "after_hours_app_usage_hrs": max(0, np.random.normal(s*4, 0.3)),
        "weekend_logins":            int(np.random.poisson(s*8)),
        "calendar_density":          np.clip(np.random.normal(0.2 + s*0.6, 0.05), 0, 1),
        "slack_response_time_min":   max(1, np.random.normal(5 + s*55, 3)),
        "task_completion_rate":      np.clip(np.random.normal(0.95 - s*0.35, 0.03), 0.1, 1),
        "pto_days_used":             int(np.random.poisson(max(0, (1-s)*2))),
        "focus_time_blocks":         max(0, np.random.normal(5 - s*3.5, 0.3)),
        "email_volume_per_day":      max(0, np.random.normal(15 + s*55, 5)),
    }

def predict_row(signals, dept, role):
    dept_enc = le_dept.transform([dept])[0] if dept in DEPTS else 0
    role_enc = le_role.transform([role])[0] if role in ROLES else 0
    row = pd.DataFrame([{**signals, "dept_enc": dept_enc, "role_enc": role_enc}])
    X = scaler.transform(row[FEATURES])
    score = float(np.clip(reg.predict(X)[0], 0, 100))
    label = le_label.inverse_transform([clf.predict(X)[0]])[0]
    shap_vals = explainer.shap_values(X)[0]
    shap_df = pd.DataFrame({"feature":FEATURES,"shap":shap_vals,"abs":np.abs(shap_vals)}).sort_values("abs",ascending=False).head(5)
    return score, label, shap_df

# ── Build demo dashboard ──────────────────────────────────────
def load_demo():
    np.random.seed(42)
    rows = []
    for emp in DEMO_EMPLOYEES:
        latest_score = emp["trend"][-1]
        sig = make_signals(latest_score)
        score, label, _ = predict_row(sig, emp["dept"], emp["role"])
        rows.append({**emp, "score": round(score,1), "label": label, "signals": sig})

    # ── Trend chart ──
    fig_trend = go.Figure()
    colors_map = {"High":"#E24B4A","Medium":"#EF9F27","Low":"#1D9E75","default":"#7F77DD"}
    weeks = [f"W{i+1}" for i in range(12)]
    for r in rows:
        col = RC.get(r["label"], "#7F77DD")
        fig_trend.add_trace(go.Scatter(
            x=weeks, y=r["trend"], mode="lines+markers",
            name=r["name"].split()[0],
            line=dict(color=col, width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{r['name']}</b><br>Week: %{{x}}<br>Score: %{{y}}<extra></extra>",
        ))
    fig_trend.add_hrect(y0=70,y1=100,fillcolor="#E24B4A",opacity=0.08,line_width=0,annotation_text="High Risk Zone",annotation_position="right")
    fig_trend.add_hrect(y0=40,y1=70,fillcolor="#EF9F27",opacity=0.08,line_width=0,annotation_text="Medium Risk Zone",annotation_position="right")
    fig_trend.update_layout(
        title="12-Week Burnout Trend — All Employees",
        xaxis_title="Week", yaxis_title="Burnout Score",
        yaxis=dict(range=[0,100]),
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(size=12),
        margin=dict(l=10,r=80,t=60,b=40),
    )
    fig_trend.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig_trend.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    # ── Department heatmap ──
    dept_scores = {}
    for r in rows:
        dept_scores.setdefault(r["dept"],[]).append(r["score"])
    dept_avg = {d: round(np.mean(v),1) for d,v in dept_scores.items()}
    dept_df = pd.DataFrame(list(dept_avg.items()), columns=["Department","Avg Score"]).sort_values("Avg Score",ascending=True)
    bar_colors = [RC["High"] if s>=70 else RC["Medium"] if s>=40 else RC["Low"] for s in dept_df["Avg Score"]]
    fig_dept = go.Figure(go.Bar(
        x=dept_df["Avg Score"], y=dept_df["Department"],
        orientation="h", marker_color=bar_colors,
        text=[f"{s}" for s in dept_df["Avg Score"]], textposition="outside",
    ))
    fig_dept.update_layout(
        title="Avg Burnout Score by Department",
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,105], title="Score"),
        margin=dict(l=10,r=60,t=50,b=30), font=dict(size=12),
    )
    fig_dept.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    # ── Risk donut ──
    counts = {"High":0,"Medium":0,"Low":0}
    for r in rows: counts[r["label"]] += 1
    fig_donut = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        hole=0.6, marker_colors=[RC["High"],RC["Medium"],RC["Low"]],
        textinfo="label+percent",
    ))
    fig_donut.add_annotation(text=f"{len(rows)}<br>Employees", x=0.5, y=0.5,
                             font_size=14, showarrow=False)
    fig_donut.update_layout(
        title="Risk Distribution", height=280,
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        margin=dict(l=10,r=10,t=50,b=10),
    )

    # ── Summary cards HTML ──
    high_risk  = [r for r in rows if r["label"]=="High"]
    med_risk   = [r for r in rows if r["label"]=="Medium"]
    avg_score  = round(np.mean([r["score"] for r in rows]),1)
    top_dept   = dept_df.iloc[-1]["Department"]

    cards_html = f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:8px;'>
      <div style='padding:16px;border-radius:12px;background:#FCEBEB;border:1px solid #E24B4A;text-align:center;'>
        <div style='font-size:32px;font-weight:700;color:#E24B4A;'>{len(high_risk)}</div>
        <div style='font-size:12px;color:#A32D2D;font-weight:500;margin-top:2px;'>HIGH RISK</div>
        <div style='font-size:11px;color:#999;margin-top:4px;'>Need immediate action</div>
      </div>
      <div style='padding:16px;border-radius:12px;background:#FAEEDA;border:1px solid #EF9F27;text-align:center;'>
        <div style='font-size:32px;font-weight:700;color:#BA7517;'>{len(med_risk)}</div>
        <div style='font-size:12px;color:#854F0B;font-weight:500;margin-top:2px;'>MEDIUM RISK</div>
        <div style='font-size:11px;color:#999;margin-top:4px;'>Monitor closely</div>
      </div>
      <div style='padding:16px;border-radius:12px;background:#E6F1FB;border:1px solid #378ADD;text-align:center;'>
        <div style='font-size:32px;font-weight:700;color:#185FA5;'>{avg_score}</div>
        <div style='font-size:12px;color:#0C447C;font-weight:500;margin-top:2px;'>AVG SCORE</div>
        <div style='font-size:11px;color:#999;margin-top:4px;'>Team average</div>
      </div>
      <div style='padding:16px;border-radius:12px;background:#FCEBEB;border:1px solid #E24B4A;text-align:center;'>
        <div style='font-size:20px;font-weight:700;color:#E24B4A;'>{top_dept}</div>
        <div style='font-size:12px;color:#A32D2D;font-weight:500;margin-top:2px;'>MOST AT RISK</div>
        <div style='font-size:11px;color:#999;margin-top:4px;'>Dept needing attention</div>
      </div>
    </div>"""

    # ── Employee table HTML ──
    table_rows = ""
    for r in sorted(rows, key=lambda x: -x["score"]):
        col  = RC[r["label"]]
        bg   = RB[r["label"]]
        bar_w = int(r["score"])
        table_rows += f"""
        <tr style='border-bottom:1px solid rgba(128,128,128,0.1);'>
          <td style='padding:10px 12px;font-weight:500;'>{r['name']}</td>
          <td style='padding:10px 12px;color:#888;font-size:13px;'>{r['dept']}</td>
          <td style='padding:10px 12px;color:#888;font-size:13px;'>{r['role']}</td>
          <td style='padding:10px 12px;'>
            <div style='display:flex;align-items:center;gap:8px;'>
              <div style='flex:1;height:6px;border-radius:3px;background:rgba(128,128,128,0.15);'>
                <div style='width:{bar_w}%;height:100%;border-radius:3px;background:{col};'></div>
              </div>
              <span style='font-weight:600;color:{col};min-width:32px;'>{r['score']}</span>
            </div>
          </td>
          <td style='padding:10px 12px;'>
            <span style='padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;
                  background:{bg};color:{col};border:1px solid {col};'>{r['label']}</span>
          </td>
        </tr>"""

    table_html = f"""
    <div style='border-radius:12px;overflow:hidden;border:1px solid rgba(128,128,128,0.15);margin-top:8px;'>
      <table style='width:100%;border-collapse:collapse;font-size:14px;'>
        <thead>
          <tr style='background:rgba(128,128,128,0.08);'>
            <th style='padding:10px 12px;text-align:left;font-weight:600;color:#888;font-size:12px;'>EMPLOYEE</th>
            <th style='padding:10px 12px;text-align:left;font-weight:600;color:#888;font-size:12px;'>DEPARTMENT</th>
            <th style='padding:10px 12px;text-align:left;font-weight:600;color:#888;font-size:12px;'>ROLE</th>
            <th style='padding:10px 12px;text-align:left;font-weight:600;color:#888;font-size:12px;'>BURNOUT SCORE</th>
            <th style='padding:10px 12px;text-align:left;font-weight:600;color:#888;font-size:12px;'>RISK</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>"""

    return cards_html, fig_trend, fig_dept, fig_donut, table_html


# ── Single employee predict ───────────────────────────────────
def predict_single(name, dept, role, typing_speed, meeting_hours, after_hours,
                   weekend_logins, calendar_density, slack_response,
                   task_completion, pto_days, focus_blocks, email_volume):
    signals = {
        "typing_speed_wpm": typing_speed,
        "meeting_hours_per_day": meeting_hours,
        "after_hours_app_usage_hrs": after_hours,
        "weekend_logins": weekend_logins,
        "calendar_density": calendar_density / 100,
        "slack_response_time_min": slack_response,
        "task_completion_rate": task_completion / 100,
        "pto_days_used": pto_days,
        "focus_time_blocks": focus_blocks,
        "email_volume_per_day": email_volume,
    }
    score, label, shap_df = predict_row(signals, dept, role)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score,1),
        title={"text": f"Burnout Risk — {name or 'Employee'}", "font":{"size":14}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1},
            "bar":{"color":RC[label],"thickness":0.25},
            "steps":[{"range":[0,40],"color":"#E1F5EE"},{"range":[40,70],"color":"#FAEEDA"},{"range":[70,100],"color":"#FCEBEB"}],
            "threshold":{"line":{"color":RC[label],"width":3},"thickness":0.8,"value":score},
        },
        number={"suffix":"/100","font":{"size":30}},
    ))
    gauge.update_layout(height=230,margin=dict(l=20,r=20,t=50,b=10),paper_bgcolor="rgba(0,0,0,0)")

    shap_colors = ["#E24B4A" if v>0 else "#1D9E75" for v in shap_df["shap"]]
    shap_fig = go.Figure(go.Bar(
        x=shap_df["shap"].round(3),
        y=shap_df["feature"].str.replace("_"," ").str.title(),
        orientation="h", marker_color=shap_colors,
        text=shap_df["shap"].round(2), textposition="outside",
    ))
    shap_fig.update_layout(
        title="Why this score? (SHAP drivers)",
        height=260, margin=dict(l=10,r=50,t=50,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_autorange="reversed", font=dict(size=12),
        xaxis_title="Impact on burnout score",
    )
    shap_fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    recs = {
        "High":   "🔴 Immediate action needed. Schedule a 1:1 this week, consider workload redistribution, and offer mental health support. Risk of resignation is high.",
        "Medium": "🟡 Monitor closely. Check in bi-weekly, review meeting load, encourage PTO usage. Intervene now to prevent escalation.",
        "Low":    "🟢 Healthy baseline. Maintain current working conditions and do a monthly check-in.",
    }
    result_html = f"""
    <div style='display:flex;gap:12px;flex-wrap:wrap;'>
      <div style='flex:1;min-width:200px;padding:20px;border-radius:12px;
           background:{RB[label]};border:2px solid {RC[label]};text-align:center;'>
        <div style='font-size:13px;color:#888;font-weight:500;letter-spacing:0.05em;'>RISK ASSESSMENT</div>
        <div style='font-size:36px;font-weight:700;color:{RC[label]};margin:6px 0;'>{label} Risk</div>
        <div style='font-size:14px;color:#666;'>Score: {score:.1f} / 100</div>
      </div>
      <div style='flex:2;min-width:250px;padding:20px;border-radius:12px;
           background:rgba(128,128,128,0.05);border:1px solid rgba(128,128,128,0.15);
           font-size:14px;line-height:1.8;color:#444;'>
        <div style='font-weight:600;margin-bottom:6px;'>Recommendation</div>
        {recs[label]}
      </div>
    </div>"""
    return gauge, shap_fig, result_html


# ── UI ────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.tab-nav button { font-size: 14px !important; font-weight: 500 !important; }
"""

with gr.Blocks(title="BurnoutRadar", css=CSS) as demo:

    gr.HTML("""
    <div style='padding:28px 0 16px;border-bottom:1px solid rgba(128,128,128,0.15);margin-bottom:20px;'>
      <div style='display:flex;align-items:center;gap:12px;'>
        <div style='font-size:28px;font-weight:700;letter-spacing:-0.5px;'>BurnoutRadar</div>
        <span style='padding:3px 10px;border-radius:20px;background:#E1F5EE;color:#0F6E56;
              font-size:12px;font-weight:600;border:1px solid #1D9E75;'>v2.0</span>
      </div>
      <div style='font-size:14px;color:#888;margin-top:4px;'>
        AI-powered early burnout detection · Predict risk 2–3 weeks before it shows · Built with XGBoost + SHAP
      </div>
    </div>""")

    with gr.Tabs():

        # ── Tab 1: Team Dashboard ──────────────────────────────
        with gr.TabItem("Team Dashboard"):
            gr.Markdown("Click **Load Demo** to instantly see a realistic team burnout analysis.")
            load_btn = gr.Button("Load Demo Team", variant="primary", size="lg")

            cards_out  = gr.HTML()
            with gr.Row():
                trend_out = gr.Plot()
            with gr.Row():
                dept_out  = gr.Plot()
                donut_out = gr.Plot()
            table_out  = gr.HTML()

            load_btn.click(load_demo, inputs=[], outputs=[cards_out, trend_out, dept_out, donut_out, table_out])

        # ── Tab 2: Single Employee ─────────────────────────────
        with gr.TabItem("Analyze Employee"):
            gr.Markdown("### Enter behavioral signals for one employee")
            with gr.Row():
                emp_name = gr.Textbox(label="Employee name", placeholder="e.g. Arjun Mehta")
                department = gr.Dropdown(DEPTS, value="Engineering", label="Department")
                role = gr.Dropdown(ROLES, value="Senior", label="Role")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Work patterns**")
                    typing_speed     = gr.Slider(20,100,value=65, step=1,   label="Typing speed (WPM)")
                    meeting_hours    = gr.Slider(0,10, value=3,  step=0.5,  label="Meeting hours / day")
                    after_hours      = gr.Slider(0,8,  value=1,  step=0.25, label="After-hours app usage (hrs)")
                    weekend_logins   = gr.Slider(0,15, value=1,  step=1,    label="Weekend logins / month")
                    calendar_density = gr.Slider(0,100,value=40, step=1,    label="Calendar density (%)")
                with gr.Column():
                    gr.Markdown("**Productivity signals**")
                    slack_response   = gr.Slider(1,120, value=10,step=1,   label="Avg Slack response time (min)")
                    task_completion  = gr.Slider(10,100,value=85,step=1,   label="Task completion rate (%)")
                    pto_days         = gr.Slider(0,20,  value=3, step=1,   label="PTO days used this quarter")
                    focus_blocks     = gr.Slider(0,8,   value=3, step=0.5, label="Deep focus blocks / day")
                    email_volume     = gr.Slider(0,100, value=25,step=1,   label="Emails received / day")

            analyze_btn = gr.Button("Analyze Burnout Risk", variant="primary", size="lg")
            result_html = gr.HTML()
            with gr.Row():
                gauge_out = gr.Plot()
                shap_out  = gr.Plot()

            analyze_btn.click(
                predict_single,
                inputs=[emp_name,department,role,typing_speed,meeting_hours,after_hours,
                        weekend_logins,calendar_density,slack_response,task_completion,
                        pto_days,focus_blocks,email_volume],
                outputs=[gauge_out,shap_out,result_html],
            )

        # ── Tab 3: How It Works ────────────────────────────────
        with gr.TabItem("How It Works"):
            gr.Markdown("""
## The problem BurnoutRadar solves

Employee burnout costs companies **$125,000–$190,000 per employee** in turnover, healthcare, and lost productivity (Gallup, 2023). The catch — by the time it's visible in performance reviews, it's already too late.

**BurnoutRadar detects it 2–3 weeks early** by analyzing behavioral signals that change before burnout becomes visible.

## Signals tracked

| Signal | Why it matters |
|---|---|
| After-hours app usage | #1 predictor — boundary erosion happens first |
| Weekend logins | Inability to disconnect = early warning |
| Task completion rate | Drops weeks before performance review flags it |
| Meeting hours / day | Cognitive overload accumulates silently |
| Slack response time | Slows as mental load increases |
| Focus blocks / day | Fragmented attention = burnout precursor |
| Calendar density | No recovery time = inevitable crash |

## Model architecture

- **XGBoost Regressor** → Burnout score (0–100)
- **XGBoost Classifier** → Risk label (Low / Medium / High)
- **SHAP explainability** → Every prediction explains itself
- Trained on 7,500+ employee behavioral records
- **MAE: 4.33 pts · R²: 0.796 · Accuracy: 88%**

## Why SHAP matters

Most AI tools are black boxes. BurnoutRadar shows HR *exactly* which signals drove each score — so managers get actionable insight, not just a number.

## Pricing
| Plan | Price | Team Size |
|---|---|---|
| Free | $0 | Up to 10 employees |
| Starter | $99 / month | Up to 100 employees |
| Growth | $299 / month | Up to 500 employees |
| Enterprise | Custom | Unlimited + API access |
""")

demo.launch()
