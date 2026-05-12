"""
Data Science Autometer — Main Streamlit Application
Automated ML Pipeline: Upload → Profile → Train → Compare → Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io

from src.data_ingestion import load_data, profile_data
from src.preprocessing import handle_missing, encode_categoricals, scale_features, split_data
from src.model_selection import get_classifiers, get_regressors, detect_task_type
from src.training import train_models
from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
    get_best_model,
    get_confusion_matrix,
)

# ─── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Science Autometer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.main-header h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin-top: 0.4rem;
}

.metric-card {
    background: linear-gradient(135deg, #1E293B, #334155);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.15);
}
.metric-card .label {
    color: #94A3B8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    color: #F1F5F9;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.2rem;
}

.best-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(59,130,246,0.12));
    border: 2px solid #6C63FF;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.best-card .title {
    color: #6C63FF;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.best-card .name {
    color: #F1F5F9;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.3rem;
}

.badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-cls {
    background: rgba(34,197,94,0.15);
    color: #22C55E;
    border: 1px solid rgba(34,197,94,0.3);
}
.badge-reg {
    background: rgba(249,115,22,0.15);
    color: #F97316;
    border: 1px solid rgba(249,115,22,0.3);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Header ────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>⚡ Data Science Autometer</h1>
    <p>Automated ML Pipeline &mdash; Upload, Train, Compare, Export</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Session State ─────────────────────────────────────────────────
for key in ["df", "results", "comparison", "best_name", "best_model",
            "task_type", "models_results", "X_test", "y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")

    upload_tab, sample_tab = st.tabs(["Upload", "Sample"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.results = None

    with sample_tab:
        sample = st.selectbox(
            "Choose sample",
            ["— Select —", "Iris (Classification)", "Titanic (Classification)", "Housing (Regression)"],
        )
        if sample != "— Select —":
            if st.button("Load Sample", width="stretch"):
                if "Iris" in sample:
                    st.session_state.df = pd.read_csv("sample_data/iris.csv")
                elif "Titanic" in sample:
                    st.session_state.df = pd.read_csv("sample_data/titanic.csv")
                else:
                    st.session_state.df = pd.read_csv("sample_data/housing.csv")
                st.session_state.results = None
                st.rerun()

    st.divider()

    run_btn = False
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### ⚙️ Configuration")
        target_col = st.selectbox("🎯 Target Column", df.columns)
        missing_strategy = st.selectbox("🔧 Missing Values", ["mean", "median", "mode", "drop"])
        scale_method = st.selectbox("📏 Scaling", ["standard", "minmax", "none"])
        test_size = st.slider("📊 Test Split", 0.1, 0.5, 0.2, 0.05)
        st.divider()
        run_btn = st.button("🚀 Run Pipeline", type="primary", width="stretch")

# ─── Main Area ─────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown(
        """
    <div style="text-align:center;padding:4rem 1rem;">
        <p style="font-size:3.5rem;margin-bottom:0.5rem;">📊</p>
        <h3 style="color:#94A3B8;">Upload a dataset or select a sample to begin</h3>
        <p style="color:#64748B;">Supports CSV and Excel files</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

df = st.session_state.df
prof = profile_data(df)

# ─── Run Pipeline ──────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running pipeline..."):
        # 1. Preprocess
        status = st.status("⏳ Running pipeline...", expanded=True)
        status.write("🔧 Handling missing values...")
        df_clean = handle_missing(df, strategy=missing_strategy)

        status.write("🏷️ Encoding categorical features...")
        df_enc, _ = encode_categoricals(df_clean, target_col)

        X = df_enc.drop(columns=[target_col])
        y = df_enc[target_col]
        task_type = detect_task_type(y)

        status.write(f"📏 Scaling features ({scale_method})...")
        X_scaled, _ = scale_features(X, method=scale_method)

        status.write(f"✂️ Splitting data ({int((1-test_size)*100)}/{int(test_size*100)})...")
        X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size)

        # 2. Train
        status.write("🤖 Training models...")
        if task_type == "classification":
            models = get_classifiers()
        else:
            models = get_regressors()
        models_results = train_models(models, X_train, y_train)

        # 3. Evaluate
        status.write("📊 Evaluating models...")
        if task_type == "classification":
            comparison = evaluate_classification(models_results, X_test, y_test)
        else:
            comparison = evaluate_regression(models_results, X_test, y_test)
        best_name, best_model = get_best_model(comparison, models_results)

        # Store in session
        st.session_state.results = True
        st.session_state.comparison = comparison
        st.session_state.best_name = best_name
        st.session_state.best_model = best_model
        st.session_state.task_type = task_type
        st.session_state.models_results = models_results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        status.update(label="✅ Pipeline complete!", state="complete")

# ─── Tabs ──────────────────────────────────────────────────────────
if st.session_state.results:
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🏆 Results", "📥 Export"])
else:
    tab1, tab2 = st.tabs(["📋 Data Preview", "📊 Data Profile"])
    tab3 = None

# ── Data Preview ───────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Rows</div>'
            f'<div class="value">{prof["shape"][0]}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Columns</div>'
            f'<div class="value">{prof["shape"][1]}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Numeric</div>'
            f'<div class="value">{len(prof["numeric_cols"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        miss = sum(prof["missing"].values())
        st.markdown(
            f'<div class="metric-card"><div class="label">Missing</div>'
            f'<div class="value">{miss}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df, width="stretch", height=400)

# ── Data Profile (only when results not ready) ────────────────────
if not st.session_state.results:
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Column Types")
            dtype_df = pd.DataFrame(
                {"Column": prof["dtypes"].keys(), "Type": prof["dtypes"].values()}
            )
            st.dataframe(dtype_df, width="stretch", hide_index=True)
        with col_b:
            st.markdown("#### Missing Values")
            miss_df = pd.DataFrame(
                {
                    "Column": prof["missing"].keys(),
                    "Count": prof["missing"].values(),
                    "Percent": [f"{v}%" for v in prof["missing_pct"].values()],
                }
            )
            st.dataframe(miss_df, width="stretch", hide_index=True)

        if prof["numeric_cols"]:
            st.markdown("#### Numeric Summary")
            st.dataframe(df[prof["numeric_cols"]].describe().round(2), width="stretch")

# ── Results ────────────────────────────────────────────────────────
if st.session_state.results and tab3 is not None:
    with tab2:
        task_type = st.session_state.task_type
        comparison = st.session_state.comparison
        best_name = st.session_state.best_name
        models_results = st.session_state.models_results

        # Task type badge
        badge_cls = "badge-cls" if task_type == "classification" else "badge-reg"
        st.markdown(
            f'<span class="badge {badge_cls}">{task_type.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Best model card
        if task_type == "classification":
            best_score = comparison.iloc[0]["Accuracy"]
            score_label = "Accuracy"
        else:
            best_score = comparison.iloc[0]["R² Score"]
            score_label = "R² Score"

        st.markdown(
            f"""
        <div class="best-card">
            <div class="title">🏆 Best Model</div>
            <div class="name">{best_name}</div>
            <div style="color:#94A3B8;margin-top:0.3rem;">{score_label}: <strong style="color:#22C55E;">{best_score}</strong></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Comparison table
        st.markdown("#### Model Comparison")
        st.dataframe(
            comparison.style.highlight_max(
                subset=[c for c in comparison.columns if c not in ["Model", "Train Time (s)"]],
                color="#6C63FF33",
            ),
            width="stretch",
            hide_index=True,
        )

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if task_type == "classification":
                metric_col = "Accuracy"
            else:
                metric_col = "R² Score"
            fig_bar = px.bar(
                comparison,
                x="Model",
                y=metric_col,
                color=metric_col,
                color_continuous_scale=["#3B82F6", "#6C63FF", "#22C55E"],
                title=f"{metric_col} by Model",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width="stretch")

        with chart_col2:
            if task_type == "classification":
                # Confusion matrix for best model
                best_model_obj = models_results[best_name]["model"]
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                cm = get_confusion_matrix(best_model_obj, X_test, y_test)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale=["#0E1117", "#6C63FF"],
                    title=f"Confusion Matrix — {best_name}",
                    template="plotly_dark",
                )
                fig_cm.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_cm, width="stretch")
            else:
                # Training time chart
                fig_time = px.bar(
                    comparison,
                    x="Model",
                    y="Train Time (s)",
                    title="Training Time Comparison",
                    template="plotly_dark",
                    color_discrete_sequence=["#6C63FF"],
                )
                fig_time.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_time, width="stretch")

        # All metrics radar chart for classification
        if task_type == "classification":
            st.markdown("#### Training Time")
            fig_time = px.bar(
                comparison,
                x="Model",
                y="Train Time (s)",
                title="Training Time (seconds)",
                template="plotly_dark",
                color_discrete_sequence=["#06B6D4"],
            )
            fig_time.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_time, width="stretch")

    # ── Export ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 📥 Download Results")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            # Download comparison CSV
            csv_buf = comparison.to_csv(index=False)
            st.download_button(
                "📄 Download Comparison (CSV)",
                csv_buf,
                file_name="model_comparison.csv",
                mime="text/csv",
                width="stretch",
            )

        with col_dl2:
            # Download best model pickle
            pkl_buf = io.BytesIO()
            pickle.dump(st.session_state.best_model, pkl_buf)
            pkl_buf.seek(0)
            st.download_button(
                "🤖 Download Best Model (.pkl)",
                pkl_buf,
                file_name=f"best_model_{best_name.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream",
                width="stretch",
            )

        st.info(
            f"**Best Model:** {st.session_state.best_name}  \n"
            f"**Task Type:** {st.session_state.task_type}  \n"
            f"Use `pickle.load(open('model.pkl', 'rb'))` to load the model."
        )
