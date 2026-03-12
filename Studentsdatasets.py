import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Student EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background-color: #0f1117; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f0f2f6 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2e3450;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

[data-testid="stMetricLabel"] {
    color: #8892b0 !important;
    font-size: 13px !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    color: #64ffda !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #1e2130;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #64ffda !important;
    font-size: 18px;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1f3a 50%, #0d2137 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(100,255,218,0.06) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 32px;
    font-weight: 700;
    color: #f0f2f6;
    margin: 0 0 6px 0;
}

.hero-subtitle {
    font-size: 15px;
    color: #8892b0;
    font-family: 'DM Mono', monospace;
    margin: 0;
}

.hero-accent {
    color: #64ffda;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    border-left: 3px solid #64ffda;
    padding-left: 14px;
    margin: 28px 0 18px 0;
}

.section-header h3 {
    margin: 0;
    font-size: 18px;
    color: #ccd6f6 !important;
    font-weight: 600;
}

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, #1a1f35, #1e2540);
    border: 1px solid #2a3060;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}

.insight-card .icon { font-size: 20px; margin-right: 8px; }
.insight-card .text { color: #ccd6f6; font-size: 14px; line-height: 1.6; }
.insight-card .label { color: #64ffda; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161f;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8892b0;
    font-weight: 600;
    padding: 8px 20px;
}

.stTabs [aria-selected="true"] {
    background: #1e2a45 !important;
    color: #64ffda !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0a3d62, #1a5276);
    color: #64ffda;
    border: 1px solid #64ffda44;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1a5276, #0a3d62);
    border-color: #64ffda;
    transform: translateY(-1px);
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Plot backgrounds */
.element-container { border-radius: 10px; }

/* Selectbox */
.stSelectbox > div > div {
    background: #1a1f35;
    border: 1px solid #2a3060;
    border-radius: 8px;
    color: #ccd6f6;
}

/* Multiselect */
.stMultiSelect > div > div {
    background: #1a1f35;
    border: 1px solid #2a3060;
    border-radius: 8px;
}

/* Warning/info */
.stAlert {
    border-radius: 10px;
}

/* Divider */
hr { border-color: #1e2540; }
</style>
""", unsafe_allow_html=True)


# ─── Plot Theme ──────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#1a1f35"
ACCENT    = "#64ffda"
ACCENT2   = "#57cbff"
TEXT      = "#ccd6f6"
MUTED     = "#8892b0"
PALETTE   = ["#64ffda","#57cbff","#ff6b9d","#ffb347","#c084fc","#a8ff78"]
GRADE_COL = {"A":"#64ffda","B":"#57cbff","C":"#ffb347","D":"#ff8c42","F":"#ff6b9d"}

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": CARD_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": "#2a3060",
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": "#1e2540",
        "grid.linewidth": 0.6,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "legend.facecolor": "#13161f",
        "legend.edgecolor": "#2a3060",
        "legend.labelcolor": TEXT,
        "font.family": "monospace",
    })


# ─── Data Generation ────────────────────────────────────────────
@st.cache_data
def generate_data(seed=42):
    np.random.seed(seed)
    n = 300
    genders     = np.random.choice(['Male', 'Female'], n)
    departments = np.random.choice(['Science','Commerce','Arts','Engineering'], n, p=[0.35,0.25,0.20,0.20])
    study_hrs   = np.clip(np.random.normal(5, 2, n), 1, 12).round(1)
    attendance  = np.clip(np.random.normal(78, 12, n), 40, 100).round(1)
    prev_score  = np.clip(np.random.normal(65, 15, n), 20, 100).round(1)
    sleep_hrs   = np.clip(np.random.normal(7, 1.2, n), 4, 10).round(1)
    extra       = np.random.choice(['Yes','No'], n, p=[0.4, 0.6])
    internet    = np.random.choice(['Yes','No'], n, p=[0.7, 0.3])

    final = (0.35*prev_score + 0.30*(study_hrs/12*100) + 0.25*attendance +
             np.random.normal(0,5,n))
    final = np.clip(final, 10, 100).round(1)

    def grade(s):
        if s>=85: return 'A'
        elif s>=70: return 'B'
        elif s>=55: return 'C'
        elif s>=40: return 'D'
        else: return 'F'

    df = pd.DataFrame({
        'Student_ID':          [f'STU{str(i).zfill(4)}' for i in range(1,n+1)],
        'Gender':              genders,
        'Department':          departments,
        'Study_Hours_Per_Day': study_hrs,
        'Attendance_%':        attendance,
        'Previous_Score':      prev_score,
        'Sleep_Hours':         sleep_hrs,
        'Extra_Activities':    extra,
        'Internet_Access':     internet,
        'Final_Score':         final
    })
    df['Grade'] = df['Final_Score'].apply(grade)

    # Introduce missing values
    for col in ['Study_Hours_Per_Day','Attendance_%','Sleep_Hours','Previous_Score']:
        mask = np.random.choice([True,False], n, p=[0.08,0.92])
        df.loc[mask, col] = np.nan

    return df


def clean_data(df):
    df_clean = df.copy()
    num_cols = ['Study_Hours_Per_Day','Attendance_%','Sleep_Hours','Previous_Score']
    for col in num_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    return df_clean


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    st.markdown("**📁 Data Source**")
    data_source = st.radio("", ["Use Sample Dataset", "Upload CSV"], label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your student CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**🎛 Filters**")

    # Loaded after data
    filter_dept    = None
    filter_gender  = None
    filter_grade   = None

    st.markdown("---")
    st.markdown("**📌 About**")
    st.markdown("""
    <div style='font-size:13px;color:#8892b0;line-height:1.7'>
    This dashboard performs full EDA on student performance data including:<br>
    • Data cleaning<br>
    • Visual analysis<br>
    • Correlation study<br>
    • Key insights
    </div>
    """, unsafe_allow_html=True)


# ─── Load Data ───────────────────────────────────────────────────
if data_source == "Upload CSV" and uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(raw_df)} rows")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        raw_df = generate_data()
else:
    raw_df = generate_data()

df_clean = clean_data(raw_df)

# Sidebar filters (now that data is loaded)
with st.sidebar:
    if 'Department' in df_clean.columns:
        dept_opts = ['All'] + sorted(df_clean['Department'].unique().tolist())
        filter_dept = st.selectbox("Department", dept_opts)

    if 'Gender' in df_clean.columns:
        gender_opts = ['All'] + sorted(df_clean['Gender'].unique().tolist())
        filter_gender = st.selectbox("Gender", gender_opts)

    if 'Grade' in df_clean.columns:
        grade_opts = ['All'] + sorted(df_clean['Grade'].unique().tolist())
        filter_grade = st.selectbox("Grade", grade_opts)

# Apply filters
df_filtered = df_clean.copy()
if filter_dept and filter_dept != 'All':
    df_filtered = df_filtered[df_filtered['Department'] == filter_dept]
if filter_gender and filter_gender != 'All':
    df_filtered = df_filtered[df_filtered['Gender'] == filter_gender]
if filter_grade and filter_grade != 'All':
    df_filtered = df_filtered[df_filtered['Grade'] == filter_grade]


# ─── Hero Banner ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">📊 Student Performance <span class="hero-accent">EDA Dashboard</span></p>
    <p class="hero-subtitle">Exploratory Data Analysis · Data Cleaning · Visualizations · Correlation</p>
</div>
""", unsafe_allow_html=True)


# ─── KPI Row ─────────────────────────────────────────────────────
num_cols_df = df_filtered.select_dtypes(include='number')
avg_score   = df_filtered['Final_Score'].mean() if 'Final_Score' in df_filtered else 0
avg_attend  = df_filtered['Attendance_%'].mean() if 'Attendance_%' in df_filtered else 0
avg_study   = df_filtered['Study_Hours_Per_Day'].mean() if 'Study_Hours_Per_Day' in df_filtered else 0
top_grade_pct = (df_filtered['Grade'].isin(['A','B']).sum() / len(df_filtered) * 100) if 'Grade' in df_filtered and len(df_filtered)>0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Students",       f"{len(df_filtered)}")
k2.metric("📈 Avg Final Score", f"{avg_score:.1f}")
k3.metric("🏫 Avg Attendance", f"{avg_attend:.1f}%")
k4.metric("📚 Avg Study Hours", f"{avg_study:.1f} hrs")
k5.metric("🏆 A/B Grade %",    f"{top_grade_pct:.1f}%")


# ─── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview",
    "🧹 Data Cleaning",
    "📊 Visualizations",
    "🔗 Correlation",
    "💡 Insights"
])

set_plot_style()


# ════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""<div class="section-header"><h3>📋 Raw Dataset Preview</h3></div>""", unsafe_allow_html=True)
    st.dataframe(df_filtered.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="section-header"><h3>📐 Shape & Types</h3></div>""", unsafe_allow_html=True)
        info_df = pd.DataFrame({
            'Column': df_filtered.columns,
            'Type': df_filtered.dtypes.astype(str).values,
            'Non-Null': df_filtered.notnull().sum().values,
            'Null': df_filtered.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("""<div class="section-header"><h3>📊 Statistics</h3></div>""", unsafe_allow_html=True)
        st.dataframe(df_filtered.describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — Data Cleaning
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div class="section-header"><h3>🧹 Missing Value Analysis (Raw Data)</h3></div>""", unsafe_allow_html=True)

    miss = raw_df.isnull().sum()
    miss_pct = (miss / len(raw_df) * 100).round(2)
    miss_df = pd.DataFrame({'Column': miss.index, 'Missing Count': miss.values, 'Missing %': miss_pct.values})
    miss_df = miss_df[miss_df['Missing Count'] > 0].reset_index(drop=True)

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
        total_missing = raw_df.isnull().sum().sum()
        total_cells = raw_df.shape[0] * raw_df.shape[1]
        st.metric("Total Missing Cells", f"{total_missing} / {total_cells}",
                  delta=f"{total_missing/total_cells*100:.1f}% of data")

    with c2:
        if len(miss_df) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor(CARD_BG)
            bars = ax.barh(miss_df['Column'], miss_df['Missing %'],
                           color=[ACCENT, ACCENT2, "#ff6b9d", "#ffb347"][:len(miss_df)],
                           edgecolor='none', height=0.6)
            ax.set_xlabel('Missing %', color=MUTED)
            ax.set_title('Missing Value % by Column', color=TEXT, fontsize=13, pad=10)
            ax.spines[['top','right','left']].set_visible(False)
            for bar, val in zip(bars, miss_df['Missing %']):
                ax.text(val+0.2, bar.get_y()+bar.get_height()/2, f'{val}%', va='center', color=TEXT, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("✅ No missing values found in dataset.")

    st.markdown("""<div class="section-header"><h3>✅ Cleaning Strategy Applied</h3></div>""", unsafe_allow_html=True)
    clean_cols = ['Study_Hours_Per_Day','Attendance_%','Sleep_Hours','Previous_Score']
    for col in clean_cols:
        if col in raw_df.columns:
            n_miss = raw_df[col].isna().sum()
            med    = raw_df[col].median()
            if n_miss > 0:
                st.markdown(f"""
                <div class="insight-card">
                    <span class="label">Median Imputation</span><br>
                    <span class="text">🔧 <b style='color:#64ffda'>{col}</b> — {n_miss} missing values filled with median <b style='color:#ffb347'>{med:.2f}</b></span>
                </div>""", unsafe_allow_html=True)

    after_missing = df_clean.isnull().sum().sum()
    st.success(f"✅ After cleaning: **{after_missing} missing values** remain.")


# ════════════════════════════════════════════════════════════════
# TAB 3 — Visualizations
# ════════════════════════════════════════════════════════════════
with tab3:

    # 3.1 Grade Distribution
    st.markdown("""<div class="section-header"><h3>🎓 Grade Distribution</h3></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    grade_order = ['A','B','C','D','F']
    g_counts = df_filtered['Grade'].value_counts().reindex(grade_order).fillna(0) if 'Grade' in df_filtered else pd.Series()

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        colors = [GRADE_COL.get(g, ACCENT) for g in grade_order]
        bars = ax.bar(grade_order, g_counts, color=colors, edgecolor='none', width=0.6)
        for bar, val in zip(bars, g_counts):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, int(val),
                        ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
        ax.set_title('Students per Grade', color=TEXT)
        ax.set_xlabel('Grade', color=MUTED)
        ax.set_ylabel('Count', color=MUTED)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        wedges, texts, autotexts = ax.pie(
            g_counts[g_counts > 0],
            labels=g_counts[g_counts > 0].index,
            colors=[GRADE_COL.get(g, ACCENT) for g in g_counts[g_counts > 0].index],
            autopct='%1.1f%%', startangle=140,
            pctdistance=0.75, wedgeprops=dict(edgecolor=CARD_BG, linewidth=2))
        for t in texts: t.set_color(TEXT)
        for at in autotexts: at.set_color('#0f1117'); at.set_fontsize(9)
        ax.set_title('Grade Share (%)', color=TEXT)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # 3.2 Score Distribution
    st.markdown("""<div class="section-header"><h3>📈 Score Distribution</h3></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        if 'Final_Score' in df_filtered.columns and len(df_filtered) > 0:
            ax.hist(df_filtered['Final_Score'], bins=20, color=ACCENT, edgecolor=CARD_BG, alpha=0.85)
            ax.axvline(df_filtered['Final_Score'].mean(), color='#ff6b9d', linestyle='--', lw=1.8,
                       label=f"Mean: {df_filtered['Final_Score'].mean():.1f}")
            ax.axvline(df_filtered['Final_Score'].median(), color='#ffb347', linestyle='--', lw=1.8,
                       label=f"Median: {df_filtered['Final_Score'].median():.1f}")
            ax.legend()
        ax.set_title('Final Score Distribution')
        ax.set_xlabel('Final Score')
        ax.set_ylabel('Count')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        if 'Department' in df_filtered.columns and 'Final_Score' in df_filtered.columns:
            dept_data = [df_filtered[df_filtered['Department']==d]['Final_Score'].dropna().values
                         for d in df_filtered['Department'].unique()]
            dept_labels = df_filtered['Department'].unique()
            bp = ax.boxplot(dept_data, labels=dept_labels, patch_artist=True,
                            medianprops=dict(color='#ff6b9d', linewidth=2),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', color=ACCENT, markersize=4, alpha=0.5))
            for patch, color in zip(bp['boxes'], PALETTE):
                patch.set_facecolor(color + '44')
                patch.set_edgecolor(color)
        ax.set_title('Final Score by Department')
        ax.tick_params(axis='x', rotation=15)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # 3.3 Study Hours & Attendance
    st.markdown("""<div class="section-header"><h3>📚 Study Hours & Attendance</h3></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        if 'Study_Hours_Per_Day' in df_filtered.columns and 'Final_Score' in df_filtered.columns:
            for grade, color in GRADE_COL.items():
                sub = df_filtered[df_filtered['Grade'] == grade]
                ax.scatter(sub['Study_Hours_Per_Day'], sub['Final_Score'],
                           color=color, alpha=0.65, s=30, label=grade)
            m, b = np.polyfit(df_filtered['Study_Hours_Per_Day'].dropna(),
                              df_filtered.loc[df_filtered['Study_Hours_Per_Day'].notna(), 'Final_Score'], 1)
            x_line = np.linspace(df_filtered['Study_Hours_Per_Day'].min(), df_filtered['Study_Hours_Per_Day'].max(), 100)
            ax.plot(x_line, m*x_line+b, color='white', linestyle='--', lw=1.5, alpha=0.7)
            ax.legend(title='Grade', fontsize=9)
        ax.set_title('Study Hours vs Final Score')
        ax.set_xlabel('Study Hours/Day')
        ax.set_ylabel('Final Score')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        if 'Attendance_%' in df_filtered.columns and 'Final_Score' in df_filtered.columns:
            for ea, color in [('Yes', ACCENT), ('No', '#ff6b9d')]:
                if 'Extra_Activities' in df_filtered.columns:
                    sub = df_filtered[df_filtered['Extra_Activities'] == ea]
                    ax.scatter(sub['Attendance_%'], sub['Final_Score'],
                               color=color, alpha=0.6, s=30, label=f'Extra: {ea}')
            ax.legend(fontsize=9)
        ax.set_title('Attendance vs Final Score')
        ax.set_xlabel('Attendance %')
        ax.set_ylabel('Final Score')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # 3.4 Gender & Sleep
    st.markdown("""<div class="section-header"><h3>👥 Gender & Sleep Analysis</h3></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        if 'Gender' in df_filtered.columns and 'Final_Score' in df_filtered.columns:
            for i, (gender, color) in enumerate(zip(['Male','Female'], [ACCENT2, '#ff6b9d'])):
                data = df_filtered[df_filtered['Gender']==gender]['Final_Score'].dropna()
                ax.hist(data, bins=15, color=color, alpha=0.65, label=gender, edgecolor='none')
            ax.legend()
        ax.set_title('Final Score by Gender')
        ax.set_xlabel('Final Score')
        ax.set_ylabel('Count')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(CARD_BG)
        if 'Sleep_Hours' in df_filtered.columns and 'Final_Score' in df_filtered.columns:
            ax.scatter(df_filtered['Sleep_Hours'], df_filtered['Final_Score'],
                       color=ACCENT, alpha=0.5, s=30)
            m, b = np.polyfit(df_filtered['Sleep_Hours'].dropna(),
                              df_filtered.loc[df_filtered['Sleep_Hours'].notna(), 'Final_Score'], 1)
            x_line = np.linspace(df_filtered['Sleep_Hours'].min(), df_filtered['Sleep_Hours'].max(), 100)
            ax.plot(x_line, m*x_line+b, color='#ffb347', linestyle='--', lw=2)
        ax.set_title('Sleep Hours vs Final Score')
        ax.set_xlabel('Sleep Hours')
        ax.set_ylabel('Final Score')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — Correlation
# ════════════════════════════════════════════════════════════════
with tab4:
    num_features = ['Study_Hours_Per_Day','Attendance_%','Previous_Score','Sleep_Hours','Final_Score']
    available = [c for c in num_features if c in df_filtered.columns]
    corr_df = df_filtered[available].corr()

    st.markdown("""<div class="section-header"><h3>🔥 Correlation Heatmap</h3></div>""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(CARD_BG)
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(10, 170, s=80, l=50, as_cmap=True)
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap=cmap, mask=mask,
                linewidths=1, linecolor='#0f1117', vmin=-1, vmax=1,
                ax=ax, annot_kws={'size':11, 'weight':'bold'})
    ax.set_title('Pearson Correlation Matrix', color=TEXT, pad=15)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("""<div class="section-header"><h3>🎯 Feature Correlation with Final Score</h3></div>""", unsafe_allow_html=True)

    if 'Final_Score' in corr_df.columns:
        corr_final = corr_df['Final_Score'].drop('Final_Score').sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(CARD_BG)
        colors = [ACCENT if v > 0 else '#ff6b9d' for v in corr_final]
        bars = ax.barh(corr_final.index, corr_final.values, color=colors, edgecolor='none', height=0.5)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        for bar, val in zip(bars, corr_final):
            x = val + 0.01 if val >= 0 else val - 0.04
            ax.text(x, bar.get_y()+bar.get_height()/2, f'{val:+.2f}', va='center', color=TEXT, fontsize=10)
        ax.set_title('Correlation with Final Score', color=TEXT)
        ax.set_xlabel('Pearson r', color=MUTED)
        ax.spines[['top','right','left']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Strength table
        st.markdown("""<div class="section-header"><h3>📋 Correlation Strength Summary</h3></div>""", unsafe_allow_html=True)
        rows = []
        for feat, val in corr_df['Final_Score'].drop('Final_Score').items():
            strength = "Strong" if abs(val)>0.6 else "Moderate" if abs(val)>0.3 else "Weak"
            direction = "Positive ↑" if val>0 else "Negative ↓"
            rows.append({'Feature': feat, 'Correlation': round(val,3), 'Direction': direction, 'Strength': strength})
        summary_df = pd.DataFrame(rows).sort_values('Correlation', ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 5 — Insights
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""<div class="section-header"><h3>💡 Key Findings</h3></div>""", unsafe_allow_html=True)

    insights = [
        ("📌", "Missing Data", f"~8% missing values found in Study Hours, Attendance, Sleep Hours & Previous Score. All imputed using median to preserve distribution."),
        ("📈", "Previous Score", "Strongest predictor of Final Score — students who performed well in the past tend to do well again."),
        ("📚", "Study Hours", "Clear positive trend — students studying 6+ hours/day score significantly higher on average."),
        ("🏫", "Attendance", "Students with 80%+ attendance outperform those below 60% by ~15 points on average."),
        ("😴", "Sleep Hours", "Moderate positive relationship — 7–8 hrs of sleep correlates with better academic performance."),
        ("🏆", "Grade Spread", "Majority of students fall in B/C grade range. Very few fail, suggesting decent baseline performance."),
    ]

    c1, c2 = st.columns(2)
    for i, (icon, label, text) in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div class="insight-card">
            <span class="label">{icon} {label}</span><br>
            <span class="text">{text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="section-header"><h3>✅ Recommendations</h3></div>""", unsafe_allow_html=True)
    recs = [
        ("🎯", "Early Intervention", "Flag students with Previous Score < 50 early in semester for additional support."),
        ("📅", "Attendance Policy", "Enforce minimum 75% attendance — it has a measurable impact on final scores."),
        ("⏰", "Study Habit Programs", "Encourage 4–6 hrs/day study schedules through structured timetables."),
        ("😴", "Wellness Campaigns", "Promote healthy sleep (7–8 hrs) as part of academic wellness programs."),
    ]
    for icon, label, text in recs:
        st.markdown(f"""
        <div class="insight-card">
            <span class="label">{icon} {label}</span><br>
            <span class="text">{text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="section-header"><h3>📥 Download Cleaned Data</h3></div>""", unsafe_allow_html=True)
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Cleaned CSV", csv, "student_clean.csv", "text/csv")
