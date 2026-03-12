import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.main { background-color: #0f1117; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #f0f2f6 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2e3450; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 13px !important; font-family: 'DM Mono', monospace !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #64ffda !important; font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stSidebar"] { background: #13161f !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] .stMarkdown h2 { color: #64ffda !important; font-size: 18px; }
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1f3a 50%, #0d2137 100%);
    border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 32px 40px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-title { font-size: 32px; font-weight: 700; color: #f0f2f6; margin: 0 0 6px 0; }
.hero-subtitle { font-size: 15px; color: #8892b0; font-family: 'DM Mono', monospace; margin: 0; }
.hero-accent { color: #64ffda; }
.section-header { display: flex; align-items: center; gap: 10px; border-left: 3px solid #64ffda; padding-left: 14px; margin: 28px 0 18px 0; }
.section-header h3 { margin: 0; font-size: 18px; color: #ccd6f6 !important; font-weight: 600; }
.insight-card { background: linear-gradient(135deg, #1a1f35, #1e2540); border: 1px solid #2a3060; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
.insight-card .text { color: #ccd6f6; font-size: 14px; line-height: 1.6; }
.insight-card .label { color: #64ffda; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }
.stTabs [data-baseweb="tab-list"] { background: #13161f; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8892b0; font-weight: 600; padding: 8px 20px; }
.stTabs [aria-selected="true"] { background: #1e2a45 !important; color: #64ffda !important; }
.stButton>button { background: linear-gradient(135deg, #0a3d62, #1a5276); color: #64ffda; border: 1px solid #64ffda44; border-radius: 8px; font-weight: 600; }
hr { border-color: #1e2540; }
</style>
""", unsafe_allow_html=True)

# ─── Plot Theme ──────────────────────────────────────────────────
DARK_BG = "#0f1117"
CARD_BG = "#1a1f35"
ACCENT  = "#64ffda"
ACCENT2 = "#57cbff"
TEXT    = "#ccd6f6"
MUTED   = "#8892b0"
PALETTE = ["#64ffda","#57cbff","#ff6b9d","#ffb347","#c084fc","#a8ff78","#f97316","#818cf8"]

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": CARD_BG, "axes.facecolor": CARD_BG,
        "axes.edgecolor": "#2a3060", "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": "#1e2540", "grid.linewidth": 0.6,
        "axes.titlecolor": TEXT, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.facecolor": "#13161f", "legend.edgecolor": "#2a3060", "legend.labelcolor": TEXT,
        "font.family": "monospace",
    })

# ─── Sample Data ─────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 300
    genders     = np.random.choice(['Male','Female'], n)
    departments = np.random.choice(['Science','Commerce','Arts','Engineering'], n, p=[0.35,0.25,0.20,0.20])
    study_hrs   = np.clip(np.random.normal(5,2,n), 1, 12).round(1)
    attendance  = np.clip(np.random.normal(78,12,n), 40, 100).round(1)
    prev_score  = np.clip(np.random.normal(65,15,n), 20, 100).round(1)
    sleep_hrs   = np.clip(np.random.normal(7,1.2,n), 4, 10).round(1)
    extra       = np.random.choice(['Yes','No'], n, p=[0.4,0.6])
    internet    = np.random.choice(['Yes','No'], n, p=[0.7,0.3])
    final = np.clip(0.35*prev_score + 0.30*(study_hrs/12*100) + 0.25*attendance + np.random.normal(0,5,n), 10, 100).round(1)
    def grade(s):
        if s>=85: return 'A'
        elif s>=70: return 'B'
        elif s>=55: return 'C'
        elif s>=40: return 'D'
        else: return 'F'
    df = pd.DataFrame({
        'Student_ID': [f'STU{str(i).zfill(4)}' for i in range(1,n+1)],
        'Gender': genders, 'Department': departments,
        'Study_Hours_Per_Day': study_hrs, 'Attendance_%': attendance,
        'Previous_Score': prev_score, 'Sleep_Hours': sleep_hrs,
        'Extra_Activities': extra, 'Internet_Access': internet, 'Final_Score': final
    })
    df['Grade'] = df['Final_Score'].apply(grade)
    for col in ['Study_Hours_Per_Day','Attendance_%','Sleep_Hours','Previous_Score']:
        mask = np.random.choice([True,False], n, p=[0.08,0.92])
        df.loc[mask, col] = np.nan
    return df

def clean_data(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include='number').columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    st.markdown("**📁 Data Source**")
    data_source = st.radio("data_source", ["Use Sample Dataset", "Upload CSV"], label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your student CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**📌 About**")
    st.markdown("<div style='font-size:13px;color:#8892b0;line-height:1.7'>EDA Dashboard:<br>• Data cleaning<br>• Visual analysis<br>• Correlation study<br>• Key insights</div>", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────
if data_source == "Upload CSV" and uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        raw_df = generate_data()
else:
    raw_df = generate_data()

df_clean = clean_data(raw_df)

# ─── Detect column types dynamically ─────────────────────────────
num_cols   = df_clean.select_dtypes(include='number').columns.tolist()
cat_cols   = df_clean.select_dtypes(include='object').columns.tolist()

# Sidebar filters — only for columns that exist
with st.sidebar:
    st.markdown("---")
    st.markdown("**🎛 Filters**")
    active_filters = {}
    # Show up to 3 categorical filters
    for col in cat_cols[:3]:
        opts = ['All'] + sorted(df_clean[col].dropna().unique().tolist())
        if len(opts) <= 15:  # only show filter if reasonable number of options
            active_filters[col] = st.selectbox(col, opts)

# Apply filters
df_filtered = df_clean.copy()
for col, val in active_filters.items():
    if val != 'All':
        df_filtered = df_filtered[df_filtered[col] == val]

# ─── Hero Banner ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">📊 Student Performance <span class="hero-accent">EDA Dashboard</span></p>
    <p class="hero-subtitle">Exploratory Data Analysis · Data Cleaning · Visualizations · Correlation</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row — fully dynamic ──────────────────────────────────────
kpi_cols = st.columns(min(len(num_cols)+1, 5))
kpi_cols[0].metric("👥 Total Students", f"{len(df_filtered)}")
for i, col in enumerate(num_cols[:4], 1):
    val = df_filtered[col].mean()
    kpi_cols[i].metric(f"📊 Avg {col[:18]}", f"{val:.1f}")

# ─── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview", "🧹 Data Cleaning", "📊 Visualizations", "🔗 Correlation", "💡 Insights"
])
set_plot_style()

# ════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><h3>📋 Dataset Preview</h3></div>', unsafe_allow_html=True)
    st.dataframe(df_filtered.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><h3>📐 Column Info</h3></div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            'Column': df_filtered.columns,
            'Type': df_filtered.dtypes.astype(str).values,
            'Non-Null': df_filtered.notnull().sum().values,
            'Null': df_filtered.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-header"><h3>📊 Statistics</h3></div>', unsafe_allow_html=True)
        st.dataframe(df_filtered.describe().round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — Data Cleaning
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><h3>🧹 Missing Value Analysis</h3></div>', unsafe_allow_html=True)
    miss = raw_df.isnull().sum()
    miss_pct = (miss / len(raw_df) * 100).round(2)
    miss_df = pd.DataFrame({'Column': miss.index, 'Missing Count': miss.values, 'Missing %': miss_pct.values})
    miss_df = miss_df[miss_df['Missing Count'] > 0].reset_index(drop=True)

    c1, c2 = st.columns([1, 1.5])
    with c1:
        if len(miss_df) > 0:
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
            st.metric("Total Missing", f"{raw_df.isnull().sum().sum()} cells",
                      delta=f"{raw_df.isnull().sum().sum()/(raw_df.shape[0]*raw_df.shape[1])*100:.1f}% of data")
        else:
            st.success("✅ No missing values in this dataset!")

    with c2:
        if len(miss_df) > 0:
            fig, ax = plt.subplots(figsize=(7, max(3, len(miss_df)*0.6)))
            fig.patch.set_facecolor(CARD_BG)
            colors_bar = PALETTE[:len(miss_df)]
            bars = ax.barh(miss_df['Column'], miss_df['Missing %'], color=colors_bar, edgecolor='none', height=0.6)
            ax.set_xlabel('Missing %', color=MUTED)
            ax.set_title('Missing Value % by Column', color=TEXT, fontsize=13)
            ax.spines[['top','right','left']].set_visible(False)
            for bar, val in zip(bars, miss_df['Missing %']):
                ax.text(val+0.2, bar.get_y()+bar.get_height()/2, f'{val}%', va='center', color=TEXT, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>✅ Cleaning Applied</h3></div>', unsafe_allow_html=True)
    filled_any = False
    for col in raw_df.select_dtypes(include='number').columns:
        n_miss = raw_df[col].isna().sum()
        if n_miss > 0:
            med = raw_df[col].median()
            st.markdown(f'<div class="insight-card"><span class="label">Median Imputation</span><br><span class="text">🔧 <b style="color:#64ffda">{col}</b> — {n_miss} missing values filled with median <b style="color:#ffb347">{med:.2f}</b></span></div>', unsafe_allow_html=True)
            filled_any = True
    if not filled_any:
        st.info("No missing values found — no cleaning needed!")
    st.success(f"✅ After cleaning: **{df_clean.isnull().sum().sum()} missing values** remain.")

# ════════════════════════════════════════════════════════════════
# TAB 3 — Visualizations (fully dynamic)
# ════════════════════════════════════════════════════════════════
with tab3:

    # --- Categorical column distributions ---
    if cat_cols:
        st.markdown('<div class="section-header"><h3>📊 Categorical Distributions</h3></div>', unsafe_allow_html=True)
        cols_to_show = [c for c in cat_cols if df_filtered[c].nunique() <= 20][:4]
        if cols_to_show:
            grid = st.columns(min(2, len(cols_to_show)))
            for i, col in enumerate(cols_to_show):
                with grid[i % 2]:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    fig.patch.set_facecolor(CARD_BG)
                    counts = df_filtered[col].value_counts()
                    bar_colors = PALETTE[:len(counts)]
                    axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor='none', width=0.6)
                    for j, v in enumerate(counts.values):
                        axes[0].text(j, v+0.5, str(v), ha='center', fontsize=9, color=TEXT)
                    axes[0].set_title(f'{col} — Count', color=TEXT)
                    axes[0].tick_params(axis='x', rotation=20)
                    axes[0].spines[['top','right']].set_visible(False)
                    axes[1].pie(counts.values, labels=counts.index, colors=bar_colors,
                                autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                                wedgeprops=dict(edgecolor=CARD_BG, linewidth=2))
                    axes[1].set_title(f'{col} — Share', color=TEXT)
                    for t in axes[1].texts: t.set_color(TEXT)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

    # --- Numeric distributions ---
    if num_cols:
        st.markdown('<div class="section-header"><h3>📈 Numeric Distributions</h3></div>', unsafe_allow_html=True)
        for i in range(0, min(len(num_cols), 6), 2):
            pair = num_cols[i:i+2]
            cols_grid = st.columns(len(pair))
            for j, col in enumerate(pair):
                with cols_grid[j]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor(CARD_BG)
                    ax.hist(df_filtered[col].dropna(), bins=20, color=PALETTE[i+j], edgecolor='none', alpha=0.85)
                    ax.axvline(df_filtered[col].mean(), color='#ff6b9d', linestyle='--', lw=1.5,
                               label=f"Mean: {df_filtered[col].mean():.1f}")
                    ax.axvline(df_filtered[col].median(), color='#ffb347', linestyle='--', lw=1.5,
                               label=f"Median: {df_filtered[col].median():.1f}")
                    ax.set_title(f'{col} Distribution', color=TEXT)
                    ax.set_xlabel(col, color=MUTED)
                    ax.legend(fontsize=8)
                    ax.spines[['top','right']].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

    # --- Numeric vs Numeric scatter ---
    if len(num_cols) >= 2:
        st.markdown('<div class="section-header"><h3>🔵 Scatter Plots</h3></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,1])
        x_col = c1.selectbox("X Axis", num_cols, index=0)
        y_col = c2.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1))
        hue_col = c3.selectbox("Color by", ["None"] + cat_cols)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(CARD_BG)
        if hue_col != "None" and hue_col in df_filtered.columns:
            cats = df_filtered[hue_col].dropna().unique()
            for k, cat in enumerate(cats):
                sub = df_filtered[df_filtered[hue_col] == cat]
                ax.scatter(sub[x_col], sub[y_col], color=PALETTE[k % len(PALETTE)],
                           alpha=0.65, s=35, label=str(cat))
            ax.legend(title=hue_col, fontsize=9)
        else:
            ax.scatter(df_filtered[x_col], df_filtered[y_col], color=ACCENT, alpha=0.6, s=35)
        # Trend line
        valid = df_filtered[[x_col, y_col]].dropna()
        if len(valid) > 2:
            m, b = np.polyfit(valid[x_col], valid[y_col], 1)
            x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
            ax.plot(x_line, m*x_line+b, color='white', linestyle='--', lw=1.5, alpha=0.7)
        ax.set_xlabel(x_col, color=MUTED)
        ax.set_ylabel(y_col, color=MUTED)
        ax.set_title(f'{x_col} vs {y_col}', color=TEXT)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # --- Box plots: numeric by category ---
    if num_cols and cat_cols:
        st.markdown('<div class="section-header"><h3>📦 Box Plots by Category</h3></div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        box_num = b1.selectbox("Numeric column", num_cols, key="box_num")
        box_cat = b2.selectbox("Category column", [c for c in cat_cols if df_filtered[c].nunique() <= 12], key="box_cat")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(CARD_BG)
        cats = df_filtered[box_cat].dropna().unique()
        data_list = [df_filtered[df_filtered[box_cat]==c][box_num].dropna().values for c in cats]
        bp = ax.boxplot(data_list, labels=cats, patch_artist=True,
                        medianprops=dict(color='#ff6b9d', linewidth=2),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker='o', color=ACCENT, markersize=4, alpha=0.5))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color + '44')
            patch.set_edgecolor(color)
        ax.set_title(f'{box_num} by {box_cat}', color=TEXT)
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — Correlation
# ════════════════════════════════════════════════════════════════
with tab4:
    if len(num_cols) >= 2:
        corr_df = df_filtered[num_cols].corr()

        st.markdown('<div class="section-header"><h3>🔥 Correlation Heatmap</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(max(7, len(num_cols)), max(5, len(num_cols)-1)))
        fig.patch.set_facecolor(CARD_BG)
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        cmap = sns.diverging_palette(10, 170, s=80, l=50, as_cmap=True)
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap=cmap, mask=mask,
                    linewidths=1, linecolor='#0f1117', vmin=-1, vmax=1,
                    ax=ax, annot_kws={'size': 10, 'weight': 'bold'})
        ax.set_title('Pearson Correlation Matrix', color=TEXT, pad=15)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Target column selector
        st.markdown('<div class="section-header"><h3>🎯 Correlation with Target Column</h3></div>', unsafe_allow_html=True)
        target = st.selectbox("Select target column", num_cols, index=len(num_cols)-1)
        corr_target = corr_df[target].drop(target).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(3, len(corr_target)*0.5)))
        fig.patch.set_facecolor(CARD_BG)
        colors_bar = [ACCENT if v > 0 else '#ff6b9d' for v in corr_target]
        bars = ax.barh(corr_target.index, corr_target.values, color=colors_bar, edgecolor='none', height=0.5)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        for bar, val in zip(bars, corr_target):
            x = val + 0.01 if val >= 0 else val - 0.06
            ax.text(x, bar.get_y()+bar.get_height()/2, f'{val:+.2f}', va='center', color=TEXT, fontsize=10)
        ax.set_title(f'Correlation with {target}', color=TEXT)
        ax.set_xlabel('Pearson r', color=MUTED)
        ax.spines[['top','right','left']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-header"><h3>📋 Correlation Summary</h3></div>', unsafe_allow_html=True)
        rows = []
        for feat, val in corr_df[target].drop(target).items():
            strength = "Strong" if abs(val)>0.6 else "Moderate" if abs(val)>0.3 else "Weak"
            direction = "Positive ↑" if val>0 else "Negative ↓"
            rows.append({'Feature': feat, 'Correlation': round(val,3), 'Direction': direction, 'Strength': strength})
        st.dataframe(pd.DataFrame(rows).sort_values('Correlation', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

# ════════════════════════════════════════════════════════════════
# TAB 5 — Insights
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><h3>💡 Auto-Generated Insights</h3></div>', unsafe_allow_html=True)

    # Dataset overview insight
    st.markdown(f'<div class="insight-card"><span class="label">📌 Dataset Overview</span><br><span class="text">Dataset has <b style="color:#64ffda">{len(df_filtered)} rows</b> and <b style="color:#64ffda">{len(df_filtered.columns)} columns</b> — {len(num_cols)} numeric, {len(cat_cols)} categorical.</span></div>', unsafe_allow_html=True)

    # Missing data insight
    total_missing = raw_df.isnull().sum().sum()
    missing_pct = total_missing / (raw_df.shape[0] * raw_df.shape[1]) * 100
    st.markdown(f'<div class="insight-card"><span class="label">🧹 Data Quality</span><br><span class="text">Found <b style="color:#ffb347">{total_missing} missing values</b> ({missing_pct:.1f}% of data). All numeric missing values were imputed using median.</span></div>', unsafe_allow_html=True)

    # Top correlation insight
    if len(num_cols) >= 2:
        corr_matrix = df_filtered[num_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        best_pair = corr_matrix.stack().idxmax()
        best_val = corr_matrix.stack().max()
        st.markdown(f'<div class="insight-card"><span class="label">🔗 Strongest Correlation</span><br><span class="text">Highest correlation is between <b style="color:#64ffda">{best_pair[0]}</b> and <b style="color:#64ffda">{best_pair[1]}</b> with r = <b style="color:#ffb347">{best_val:.2f}</b>.</span></div>', unsafe_allow_html=True)

    # Numeric summary insights
    for col in num_cols[:4]:
        mean_val = df_filtered[col].mean()
        std_val  = df_filtered[col].std()
        st.markdown(f'<div class="insight-card"><span class="label">📊 {col}</span><br><span class="text">Mean = <b style="color:#64ffda">{mean_val:.2f}</b>, Std = <b style="color:#57cbff">{std_val:.2f}</b>, Range = [{df_filtered[col].min():.1f} – {df_filtered[col].max():.1f}]</span></div>', unsafe_allow_html=True)

    # Categorical insights
    for col in cat_cols[:2]:
        top_val = df_filtered[col].value_counts().idxmax()
        top_pct = df_filtered[col].value_counts(normalize=True).max() * 100
        st.markdown(f'<div class="insight-card"><span class="label">🏷️ {col}</span><br><span class="text">Most common value: <b style="color:#64ffda">{top_val}</b> ({top_pct:.1f}% of records). {df_filtered[col].nunique()} unique categories.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h3>📥 Download Cleaned Data</h3></div>', unsafe_allow_html=True)
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Cleaned CSV", csv, "student_clean.csv", "text/csv")
