import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background-color: #080c14; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1,h2,h3 { color: #e2e8f0 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%) !important;
    border-right: 1px solid #1e3a5f;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b2a, #112240);
    border-radius: 14px; border: 1px solid #1e3a5f;
    padding: 18px 22px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; font-family: 'JetBrains Mono', monospace !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 28px !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

.hero {
    background: linear-gradient(135deg, #080c14 0%, #0d1b2a 40%, #112240 100%);
    border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 36px 44px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: -30%; right: -5%;
    width: 500px; height: 500px; border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%);
}
.hero-badge { background: #1e3a5f; color: #38bdf8; font-size: 11px; font-family: 'JetBrains Mono', monospace; padding: 4px 12px; border-radius: 20px; display: inline-block; margin-bottom: 14px; letter-spacing: 0.1em; }
.hero-title { font-size: 34px; font-weight: 800; color: #f1f5f9; margin: 0 0 8px 0; line-height: 1.2; }
.hero-accent { color: #38bdf8; }
.hero-sub { font-size: 14px; color: #64748b; font-family: 'JetBrains Mono', monospace; margin: 0; }

.risk-card {
    border-radius: 14px; padding: 20px 24px; margin-bottom: 12px;
    border-left: 4px solid; position: relative; overflow: hidden;
}
.risk-high   { background: linear-gradient(135deg, #1a0a0a, #2d1010); border-color: #ef4444; }
.risk-medium { background: linear-gradient(135deg, #1a1200, #2d2000); border-color: #f59e0b; }
.risk-low    { background: linear-gradient(135deg, #0a1a0e, #102d18); border-color: #22c55e; }

.risk-name  { font-size: 15px; font-weight: 700; color: #f1f5f9; }
.risk-score { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.risk-high   .risk-score { color: #ef4444; }
.risk-medium .risk-score { color: #f59e0b; }
.risk-low    .risk-score { color: #22c55e; }

.section-hdr {
    border-left: 3px solid #38bdf8; padding-left: 14px;
    margin: 30px 0 18px 0;
}
.section-hdr h3 { margin: 0; font-size: 17px; color: #cbd5e1 !important; font-weight: 700; }

.info-card {
    background: linear-gradient(135deg, #0d1b2a, #112240);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 10px;
}
.info-card .label { color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
.info-card .text  { color: #94a3b8; font-size: 13px; line-height: 1.7; margin-top: 4px; }

.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-radius: 10px; gap: 3px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-weight: 700; padding: 8px 18px; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #112240 !important; color: #38bdf8 !important; }

.stButton>button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; font-size: 14px; padding: 10px 24px;
    transition: all 0.2s;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(14,165,233,0.3); }
hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ─── Plot Style ──────────────────────────────────────────────────
BG    = "#080c14"
CARD  = "#0d1b2a"
BLUE  = "#38bdf8"
GREEN = "#22c55e"
RED   = "#ef4444"
AMBER = "#f59e0b"
TEXT  = "#e2e8f0"
MUTED = "#64748b"
PAL   = [BLUE, GREEN, AMBER, RED, "#a78bfa", "#fb923c"]

def plot_style():
    plt.rcParams.update({
        "figure.facecolor": CARD, "axes.facecolor": CARD,
        "axes.edgecolor": "#1e3a5f", "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": "#112240", "grid.linewidth": 0.6,
        "axes.titlecolor": TEXT, "axes.titlesize": 13,
        "legend.facecolor": "#080c14", "legend.edgecolor": "#1e3a5f",
        "legend.labelcolor": TEXT, "font.family": "monospace",
    })

# ─── Data Generator ──────────────────────────────────────────────
@st.cache_data
def generate_data(n=500):
    np.random.seed(42)
    attendance  = np.clip(np.random.normal(72, 18, n), 20, 100).round(1)
    study_hrs   = np.clip(np.random.normal(4.5, 2, n), 0.5, 12).round(1)
    prev_score  = np.clip(np.random.normal(62, 18, n), 15, 100).round(1)
    sleep_hrs   = np.clip(np.random.normal(6.8, 1.5, n), 3, 10).round(1)
    family_inc  = np.random.choice(['Low','Medium','High'], n, p=[0.35,0.45,0.20])
    part_time   = np.random.choice([0,1], n, p=[0.6,0.4])
    distance_km = np.clip(np.random.exponential(15, n), 1, 80).round(1)
    gender      = np.random.choice(['Male','Female'], n)
    department  = np.random.choice(['Science','Commerce','Arts','Engineering'], n)
    counseling  = np.random.choice([0,1], n, p=[0.7,0.3])
    parent_edu  = np.random.choice(['None','School','Graduate','Post-Graduate'], n, p=[0.2,0.35,0.30,0.15])

    # Dropout probability based on risk factors
    risk = (
        (100 - attendance) * 0.03 +
        (10 - study_hrs)   * 0.05 +
        (70 - prev_score)  * 0.02 +
        (part_time)        * 0.15 +
        (distance_km / 80) * 0.10 +
        (family_inc == 'Low').astype(int) * 0.20 +
        np.random.normal(0, 0.1, n)
    )
    dropout_prob = 1 / (1 + np.exp(-risk + 1.5))
    dropout = (dropout_prob > 0.5).astype(int)

    df = pd.DataFrame({
        'Student_ID':    [f'STU{str(i).zfill(4)}' for i in range(1,n+1)],
        'Gender':        gender,
        'Department':    department,
        'Attendance_%':  attendance,
        'Study_Hours':   study_hrs,
        'Previous_Score':prev_score,
        'Sleep_Hours':   sleep_hrs,
        'Family_Income': family_inc,
        'Part_Time_Job': part_time,
        'Distance_km':   distance_km,
        'Counseling':    counseling,
        'Parent_Education': parent_edu,
        'Dropout':       dropout
    })
    return df

# ─── Model Training ──────────────────────────────────────────────
@st.cache_resource
def train_model(df, model_type='Random Forest'):
    features = ['Attendance_%','Study_Hours','Previous_Score','Sleep_Hours',
                'Part_Time_Job','Distance_km','Counseling']
    cat_features = ['Gender','Department','Family_Income','Parent_Education']

    df_model = df.copy()
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df_model[col+'_enc'] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
        features.append(col+'_enc')

    X = df_model[features]
    y = df_model['Dropout']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    else:
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:,1]
        feat_imp = pd.Series(np.abs(model.coef_[0]), index=features).sort_values(ascending=False)
        scaler_used = scaler

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        'model': model, 'scaler': scaler, 'encoders': encoders,
        'features': features, 'cat_features': cat_features,
        'auc': auc, 'report': report, 'cm': cm,
        'fpr': fpr, 'tpr': tpr, 'feat_imp': feat_imp,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
        'model_type': model_type
    }

def predict_student(model_data, student_dict):
    features = model_data['features']
    cat_features = model_data['cat_features']
    encoders = model_data['encoders']

    row = {}
    for f in features:
        if f.endswith('_enc'):
            col = f.replace('_enc','')
            le = encoders[col]
            val = student_dict.get(col, le.classes_[0])
            if val in le.classes_:
                row[f] = le.transform([val])[0]
            else:
                row[f] = 0
        else:
            row[f] = student_dict.get(f, 0)

    X = pd.DataFrame([row])[features]
    model = model_data['model']
    model_type = model_data['model_type']

    if model_type == 'Logistic Regression':
        X_sc = model_data['scaler'].transform(X)
        prob = model.predict_proba(X_sc)[0][1]
    else:
        prob = model.predict_proba(X)[0][1]

    if prob >= 0.65:
        risk = 'HIGH'; color = RED; emoji = '🔴'
    elif prob >= 0.35:
        risk = 'MEDIUM'; color = AMBER; emoji = '🟡'
    else:
        risk = 'LOW'; color = GREEN; emoji = '🟢'

    return prob, risk, color, emoji

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 Controls")
    st.markdown("---")
    st.markdown("**📁 Data**")
    data_src = st.radio("src", ["Use Sample Data","Upload CSV"], label_visibility="collapsed")
    uploaded = None
    if data_src == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**🤖 Model**")
    model_choice = st.selectbox("Algorithm", ["Random Forest","Gradient Boosting","Logistic Regression"])
    st.markdown("---")
    st.markdown("**🎛 Filters**")

# ─── Load Data ───────────────────────────────────────────────────
if data_src == "Upload CSV" and uploaded:
    try:
        raw_df = pd.read_csv(uploaded)
        # Check if Dropout column exists
        if 'Dropout' not in raw_df.columns:
            st.sidebar.warning("⚠️ No 'Dropout' column found. Using sample data.")
            raw_df = generate_data()
        else:
            st.sidebar.success(f"✅ {len(raw_df)} rows loaded")
    except:
        raw_df = generate_data()
else:
    raw_df = generate_data()

# Sidebar filters
with st.sidebar:
    dept_opts = ['All'] + sorted(raw_df['Department'].unique().tolist()) if 'Department' in raw_df.columns else ['All']
    sel_dept  = st.selectbox("Department", dept_opts)
    risk_filter = st.selectbox("Show Risk Level", ["All","HIGH","MEDIUM","LOW"])

df = raw_df.copy()
if sel_dept != 'All' and 'Department' in df.columns:
    df = df[df['Department'] == sel_dept]

# ─── Train Model ─────────────────────────────────────────────────
if 'Dropout' in df.columns and len(df) >= 50:
    model_data = train_model(df, model_choice)
    model_ready = True
else:
    model_ready = False

plot_style()

# ─── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🚨 EARLY WARNING SYSTEM v1.0</div>
    <p class="hero-title">Student <span class="hero-accent">Dropout</span> Predictor</p>
    <p class="hero-sub">ML-powered · Real-time risk scoring · Actionable alerts</p>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ────────────────────────────────────────────────────────
total = len(df)
dropout_count = df['Dropout'].sum() if 'Dropout' in df.columns else 0
dropout_pct   = dropout_count / total * 100 if total > 0 else 0
avg_attend    = df['Attendance_%'].mean() if 'Attendance_%' in df.columns else 0
avg_score     = df['Previous_Score'].mean() if 'Previous_Score' in df.columns else 0

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("👥 Total Students", f"{total}")
k2.metric("🚨 Dropout Cases",  f"{int(dropout_count)}", delta=f"{dropout_pct:.1f}% rate", delta_color="inverse")
k3.metric("📊 Model AUC",      f"{model_data['auc']:.3f}" if model_ready else "N/A")
k4.metric("🏫 Avg Attendance", f"{avg_attend:.1f}%")
k5.metric("📈 Avg Score",      f"{avg_score:.1f}")

# ─── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Risk Alerts", "🔮 Predict Student", "📊 Model Performance", "📈 EDA", "💡 Action Plan"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Risk Alerts
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hdr"><h3>🚨 At-Risk Student Alerts</h3></div>', unsafe_allow_html=True)

    if model_ready:
        # Predict all students
        results = []
        num_features = ['Attendance_%','Study_Hours','Previous_Score','Sleep_Hours',
                        'Part_Time_Job','Distance_km','Counseling']
        cat_features = ['Gender','Department','Family_Income','Parent_Education']

        for _, row in df.iterrows():
            student_dict = {f: row[f] for f in num_features + cat_features if f in row.index}
            prob, risk, color, emoji = predict_student(model_data, student_dict)
            results.append({
                'Student_ID': row.get('Student_ID','N/A'),
                'Department': row.get('Department','N/A'),
                'Dropout_Prob': round(prob*100, 1),
                'Risk_Level': risk,
                'Attendance_%': row.get('Attendance_%', 0),
                'Previous_Score': row.get('Previous_Score', 0),
                'Actual_Dropout': int(row.get('Dropout', 0))
            })

        results_df = pd.DataFrame(results).sort_values('Dropout_Prob', ascending=False)

        # Filter by risk
        if risk_filter != 'All':
            results_df = results_df[results_df['Risk_Level'] == risk_filter]

        # Summary cards
        high   = (results_df['Risk_Level']=='HIGH').sum()
        medium = (results_df['Risk_Level']=='MEDIUM').sum()
        low    = (results_df['Risk_Level']=='LOW').sum()

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f'<div class="risk-card risk-high"><div style="font-size:28px;font-weight:800;color:#ef4444">{high}</div><div style="color:#94a3b8;font-size:13px">HIGH RISK students</div></div>', unsafe_allow_html=True)
        rc2.markdown(f'<div class="risk-card risk-medium"><div style="font-size:28px;font-weight:800;color:#f59e0b">{medium}</div><div style="color:#94a3b8;font-size:13px">MEDIUM RISK students</div></div>', unsafe_allow_html=True)
        rc3.markdown(f'<div class="risk-card risk-low"><div style="font-size:28px;font-weight:800;color:#22c55e">{low}</div><div style="color:#94a3b8;font-size:13px">LOW RISK students</div></div>', unsafe_allow_html=True)

        # Top 15 high risk
        st.markdown('<div class="section-hdr"><h3>🔴 Top At-Risk Students</h3></div>', unsafe_allow_html=True)
        top_risk = results_df[results_df['Risk_Level']=='HIGH'].head(15)

        for _, r in top_risk.iterrows():
            prob_val = r['Dropout_Prob']
            bar_width = prob_val
            st.markdown(f"""
            <div class="risk-card risk-high">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <span class="risk-name">🔴 {r['Student_ID']}</span>
                        <span style="color:#475569;font-size:12px;margin-left:10px">{r['Department']}</span>
                    </div>
                    <span class="risk-score" style="font-size:18px;font-weight:800">{prob_val}% risk</span>
                </div>
                <div style="background:#2d1010;border-radius:4px;height:6px;margin-top:10px">
                    <div style="background:#ef4444;width:{bar_width}%;height:6px;border-radius:4px"></div>
                </div>
                <div style="display:flex;gap:16px;margin-top:8px;font-size:12px;color:#64748b;font-family:monospace">
                    <span>📅 Attend: {r['Attendance_%']:.0f}%</span>
                    <span>📊 Score: {r['Previous_Score']:.0f}</span>
                    <span>✅ Actual: {'Dropped' if r['Actual_Dropout']==1 else 'Enrolled'}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-hdr"><h3>📋 Full Risk Table</h3></div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Download
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Risk Report", csv, "dropout_risk_report.csv", "text/csv")

# ════════════════════════════════════════════════════════════════
# TAB 2 — Predict Individual Student
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr"><h3>🔮 Predict Individual Student Risk</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><span class="label">How it works</span><div class="text">Fill in the student details below and click Predict. The ML model will calculate their dropout risk probability in real-time.</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        attendance  = st.slider("Attendance %", 20, 100, 70)
        study_hrs   = st.slider("Study Hours/Day", 0.5, 12.0, 4.0, 0.5)
        prev_score  = st.slider("Previous Score", 15, 100, 60)
        sleep_hrs   = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5)
    with c2:
        gender      = st.selectbox("Gender", ["Male","Female"])
        department  = st.selectbox("Department", ["Science","Commerce","Arts","Engineering"])
        family_inc  = st.selectbox("Family Income", ["Low","Medium","High"])
        parent_edu  = st.selectbox("Parent Education", ["None","School","Graduate","Post-Graduate"])
    with c3:
        part_time   = st.selectbox("Part-Time Job", ["No","Yes"])
        distance    = st.slider("Distance from College (km)", 1, 80, 10)
        counseling  = st.selectbox("Attended Counseling", ["No","Yes"])

    if st.button("🔮 Predict Dropout Risk", use_container_width=True):
        if model_ready:
            student = {
                'Attendance_%': attendance, 'Study_Hours': study_hrs,
                'Previous_Score': prev_score, 'Sleep_Hours': sleep_hrs,
                'Part_Time_Job': 1 if part_time=="Yes" else 0,
                'Distance_km': distance,
                'Counseling': 1 if counseling=="Yes" else 0,
                'Gender': gender, 'Department': department,
                'Family_Income': family_inc, 'Parent_Education': parent_edu
            }
            prob, risk, color, emoji = predict_student(model_data, student)

            risk_class = {'HIGH':'risk-high','MEDIUM':'risk-medium','LOW':'risk-low'}[risk]
            risk_msg   = {
                'HIGH':   "⚠️ This student is at serious risk of dropping out. Immediate intervention recommended.",
                'MEDIUM': "⚡ This student shows moderate risk factors. Monitor closely and offer support.",
                'LOW':    "✅ This student appears stable. Continue regular check-ins."
            }[risk]

            st.markdown(f"""
            <div class="risk-card {risk_class}" style="margin-top:20px;padding:28px">
                <div style="font-size:48px;margin-bottom:8px">{emoji}</div>
                <div style="font-size:22px;font-weight:800;color:#f1f5f9">Dropout Risk: {prob*100:.1f}%</div>
                <div style="font-size:16px;font-weight:700;color:{color};margin:6px 0">{risk} RISK</div>
                <div style="background:rgba(0,0,0,0.3);border-radius:6px;height:10px;margin:14px 0">
                    <div style="background:{color};width:{prob*100:.0f}%;height:10px;border-radius:6px;transition:width 0.5s"></div>
                </div>
                <div style="color:#94a3b8;font-size:13px;line-height:1.6">{risk_msg}</div>
            </div>
            """, unsafe_allow_html=True)

            # Suggestions
            st.markdown('<div class="section-hdr"><h3>📋 Recommended Actions</h3></div>', unsafe_allow_html=True)
            suggestions = []
            if attendance < 75:  suggestions.append(("📅", "Low Attendance", "Attendance is below 75%. Reach out to understand barriers and enforce attendance support."))
            if study_hrs < 3:    suggestions.append(("📚", "Low Study Hours", "Study hours are very low. Recommend structured study timetables or peer groups."))
            if prev_score < 50:  suggestions.append(("📊", "Low Scores", "Previous scores indicate academic difficulty. Assign a mentor or tutor."))
            if part_time == "Yes": suggestions.append(("💼", "Part-Time Job", "Working part-time can impact academics. Check if financial aid is available."))
            if family_inc == "Low": suggestions.append(("💰", "Financial Risk", "Low family income is a major dropout driver. Connect with scholarship/financial aid programs."))
            if counseling == "No": suggestions.append(("🧠", "No Counseling", "Student has not attended counseling. Strongly recommend a session."))
            if distance > 40:    suggestions.append(("🚌", "Long Distance", "Long commute can cause fatigue and absenteeism. Explore hostel/transport options."))

            if suggestions:
                for icon, label, text in suggestions:
                    st.markdown(f'<div class="info-card"><span class="label">{icon} {label}</span><div class="text">{text}</div></div>', unsafe_allow_html=True)
            else:
                st.success("✅ No major risk factors detected. Keep up the good work!")

# ════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════
with tab3:
    if model_ready:
        st.markdown('<div class="section-hdr"><h3>📊 Model Performance Metrics</h3></div>', unsafe_allow_html=True)

        rep = model_data['report']
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("🎯 AUC-ROC",   f"{model_data['auc']:.3f}")
        m2.metric("✅ Accuracy",   f"{rep['accuracy']*100:.1f}%")
        m3.metric("🔍 Precision",  f"{rep['1']['precision']*100:.1f}%")
        m4.metric("📡 Recall",     f"{rep['1']['recall']*100:.1f}%")

        c1, c2 = st.columns(2)

        with c1:
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(CARD)
            cm = model_data['cm']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Enrolled','Dropout'], yticklabels=['Enrolled','Dropout'],
                        linewidths=2, linecolor=BG, annot_kws={'size':14,'weight':'bold'})
            ax.set_title('Confusion Matrix', color=TEXT, pad=12)
            ax.set_xlabel('Predicted', color=MUTED)
            ax.set_ylabel('Actual', color=MUTED)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with c2:
            # ROC Curve
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(CARD)
            ax.plot(model_data['fpr'], model_data['tpr'], color=BLUE, lw=2,
                    label=f"AUC = {model_data['auc']:.3f}")
            ax.plot([0,1],[0,1], color=MUTED, linestyle='--', lw=1)
            ax.fill_between(model_data['fpr'], model_data['tpr'], alpha=0.1, color=BLUE)
            ax.set_title('ROC Curve', color=TEXT)
            ax.set_xlabel('False Positive Rate', color=MUTED)
            ax.set_ylabel('True Positive Rate', color=MUTED)
            ax.legend(fontsize=11)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Feature Importance
        st.markdown('<div class="section-hdr"><h3>🏆 Feature Importance</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(CARD)
        fi = model_data['feat_imp'].head(10)
        colors_fi = [BLUE if i < 3 else MUTED for i in range(len(fi))]
        bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors_fi[::-1], edgecolor='none', height=0.6)
        ax.set_title('Top Features Driving Dropout Risk', color=TEXT)
        ax.set_xlabel('Importance Score', color=MUTED)
        ax.spines[['top','right','left']].set_visible(False)
        for bar, val in zip(bars, fi.values[::-1]):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', color=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — EDA
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr"><h3>📈 Dropout vs Key Factors</h3></div>', unsafe_allow_html=True)

    num_features = ['Attendance_%','Study_Hours','Previous_Score','Sleep_Hours','Distance_km']
    available = [f for f in num_features if f in df.columns]

    for i in range(0, len(available), 2):
        pair = available[i:i+2]
        cols = st.columns(len(pair))
        for j, feat in enumerate(pair):
            with cols[j]:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor(CARD)
                if 'Dropout' in df.columns:
                    for dropout_val, color, label in [(0, GREEN, 'Enrolled'), (1, RED, 'Dropped Out')]:
                        data = df[df['Dropout']==dropout_val][feat].dropna()
                        ax.hist(data, bins=20, color=color, alpha=0.65, label=label, edgecolor='none')
                    ax.legend(fontsize=9)
                else:
                    ax.hist(df[feat].dropna(), bins=20, color=BLUE, alpha=0.8, edgecolor='none')
                ax.set_title(f'{feat} Distribution', color=TEXT)
                ax.set_xlabel(feat, color=MUTED)
                ax.spines[['top','right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

    # Dropout by category
    if 'Department' in df.columns and 'Dropout' in df.columns:
        st.markdown('<div class="section-hdr"><h3>🏢 Dropout Rate by Department</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(CARD)
        dept_dropout = df.groupby('Department')['Dropout'].mean() * 100
        colors_d = [RED if v > 40 else AMBER if v > 25 else GREEN for v in dept_dropout.values]
        bars = ax.bar(dept_dropout.index, dept_dropout.values, color=colors_d, edgecolor='none', width=0.6)
        for bar, val in zip(bars, dept_dropout.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%',
                    ha='center', fontsize=11, color=TEXT, fontweight='bold')
        ax.set_title('Dropout Rate by Department (%)', color=TEXT)
        ax.set_ylabel('Dropout %', color=MUTED)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — Action Plan
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr"><h3>💡 Institutional Action Plan</h3></div>', unsafe_allow_html=True)

    actions = [
        ("🔴", "Immediate Intervention (HIGH Risk)", [
            "Schedule personal counseling session within 48 hours",
            "Assign a dedicated faculty mentor",
            "Check for financial difficulties — connect with scholarship cell",
            "Create customized attendance recovery plan",
            "Notify parents/guardians if student is under 18"
        ]),
        ("🟡", "Monitoring Plan (MEDIUM Risk)", [
            "Weekly check-in with class coordinator",
            "Recommend peer study groups",
            "Offer remedial classes for low-scoring subjects",
            "Monitor attendance trends weekly",
            "Encourage participation in campus activities"
        ]),
        ("🟢", "Preventive Measures (LOW Risk)", [
            "Monthly progress reviews",
            "Career guidance sessions",
            "Keep engagement high through workshops & events",
            "Recognize academic achievements to boost motivation"
        ]),
    ]

    for emoji, title, points in actions:
        risk_class = {'🔴':'risk-high','🟡':'risk-medium','🟢':'risk-low'}[emoji]
        points_html = ''.join([f'<li style="color:#94a3b8;font-size:13px;margin:5px 0">{p}</li>' for p in points])
        st.markdown(f"""
        <div class="risk-card {risk_class}" style="padding:22px">
            <div style="font-size:16px;font-weight:800;color:#f1f5f9;margin-bottom:12px">{emoji} {title}</div>
            <ul style="margin:0;padding-left:18px">{points_html}</ul>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr"><h3>📊 Key Risk Factors Summary</h3></div>', unsafe_allow_html=True)
    factors = [
        ("📅", "Attendance < 75%", "Strongest predictor of dropout. Track weekly."),
        ("💰", "Low Family Income", "Financial stress is a top cause. Scholarship awareness is key."),
        ("💼", "Part-Time Employment", "Balancing work & study increases dropout risk by ~30%."),
        ("📊", "Low Previous Scores", "Academic failure compounds over time. Early tutoring helps."),
        ("🚌", "Long Commute > 40km", "Daily travel fatigue leads to chronic absenteeism."),
        ("🧠", "No Counseling Access", "Students without mental health support are more vulnerable."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, label, text) in enumerate(factors):
        col = c1 if i%2==0 else c2
        col.markdown(f'<div class="info-card"><span class="label">{icon} {label}</span><div class="text">{text}</div></div>', unsafe_allow_html=True)
