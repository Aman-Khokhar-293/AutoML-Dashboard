"""
Data Science Autometer — Main Streamlit Application
Automated ML Pipeline: Upload → Profile → Train → Compare → Export
Supports manual Classification / Regression task-type selection.
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

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.main-header h1 { color: white; font-size: 2.4rem; font-weight: 800; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin-top: 0.4rem; }

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
    color: #94A3B8; font-size: 0.8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.metric-card .value { color: #F1F5F9; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }

.best-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(59,130,246,0.12));
    border: 2px solid #6C63FF;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.best-card .title { color: #6C63FF; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.best-card .name  { color: #F1F5F9; font-size: 1.6rem; font-weight: 700; margin-top: 0.3rem; }

.badge { display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-cls { background: rgba(34,197,94,0.15);  color: #22C55E; border: 1px solid rgba(34,197,94,0.3); }
.badge-reg { background: rgba(249,115,22,0.15); color: #F97316; border: 1px solid rgba(249,115,22,0.3); }

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# Helper Rendering Functions (defined first so they can be called)
# ═══════════════════════════════════════════════════════════════════

def _chart_layout():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )


def render_classification_charts(comparison, best_name, models_results, X_test, y_test):
    """Render all charts for a classification result."""
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            comparison, x="Model", y="Accuracy", color="Accuracy",
            color_continuous_scale=["#3B82F6", "#6C63FF", "#22C55E"],
            title="Accuracy by Model", template="plotly_dark",
        )
        fig.update_layout(**_chart_layout(), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        best_model_obj = models_results[best_name]["model"]
        y_pred_best = best_model_obj.predict(X_test)
        # Use the union of actual + predicted classes — same as sklearn's confusion_matrix
        all_classes = sorted(set(list(y_test.values) + list(y_pred_best)))
        labels = [str(l) for l in all_classes]
        cm = get_confusion_matrix(best_model_obj, X_test, y_test)
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale=["#0E1117", "#6C63FF"],
            title=f"Confusion Matrix — {best_name}",
            template="plotly_dark",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(**_chart_layout())
        st.plotly_chart(fig_cm, use_container_width=True)

    # Multi-metric grouped bar
    st.markdown("#### 📈 All Metrics — All Models")
    metric_cols = [c for c in comparison.columns if c not in ["Model", "Train Time (s)"]]
    melted = comparison.melt(id_vars=["Model"], value_vars=metric_cols,
                             var_name="Metric", value_name="Score")
    fig_multi = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        template="plotly_dark", title="Multi-Metric Comparison",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_multi.update_layout(**_chart_layout())
    st.plotly_chart(fig_multi, use_container_width=True)

    st.markdown("#### ⏱️ Training Time")
    fig_time = px.bar(
        comparison, x="Model", y="Train Time (s)",
        title="Training Time (seconds)", template="plotly_dark",
        color_discrete_sequence=["#06B6D4"],
    )
    fig_time.update_layout(**_chart_layout())
    st.plotly_chart(fig_time, use_container_width=True)


def render_regression_charts(comparison, best_name, models_results, X_test, y_test):
    """Render all charts for a regression result."""
    best_model_obj = models_results[best_name]["model"]
    y_pred = best_model_obj.predict(X_test)

    c1, c2 = st.columns(2)

    with c1:
        fig_r2 = px.bar(
            comparison, x="Model", y="R² Score", color="R² Score",
            color_continuous_scale=["#3B82F6", "#6C63FF", "#22C55E"],
            title="R² Score by Model (higher is better)", template="plotly_dark",
        )
        fig_r2.update_layout(**_chart_layout(), showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_mae = px.bar(
            comparison, x="Model", y="MAE", color="MAE",
            color_continuous_scale=["#22C55E", "#F97316", "#EF4444"],
            title="MAE by Model (lower is better)", template="plotly_dark",
        )
        fig_mae.update_layout(**_chart_layout(), showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    # Actual vs Predicted
    st.markdown(f"#### 🎯 Actual vs Predicted — {best_name}")
    all_vals = np.concatenate([y_test.values, y_pred])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig_sc = px.scatter(
        scatter_df, x="Actual", y="Predicted", opacity=0.65,
        template="plotly_dark",
        title=f"Actual vs Predicted ({best_name})",
        color_discrete_sequence=["#6C63FF"],
    )
    fig_sc.add_shape(
        type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
        line=dict(color="#22C55E", width=2, dash="dash"),
    )
    fig_sc.update_layout(**_chart_layout())
    st.plotly_chart(fig_sc, use_container_width=True)

    # Residuals
    residuals = y_test.values - y_pred
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("#### 📉 Residuals vs Predicted")
        res_df = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})
        fig_res = px.scatter(
            res_df, x="Predicted", y="Residual", opacity=0.65,
            template="plotly_dark", title="Residuals vs Predicted",
            color_discrete_sequence=["#F97316"],
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="#22C55E", line_width=2)
        fig_res.update_layout(**_chart_layout())
        st.plotly_chart(fig_res, use_container_width=True)

    with r2:
        st.markdown("#### 📊 Residual Distribution")
        fig_hist = px.histogram(
            x=residuals, nbins=30,
            template="plotly_dark", title="Residual Distribution",
            color_discrete_sequence=["#3B82F6"],
            labels={"x": "Residual"},
        )
        fig_hist.update_layout(**_chart_layout(), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### ⏱️ Training Time")
    fig_time = px.bar(
        comparison, x="Model", y="Train Time (s)",
        title="Training Time (seconds)", template="plotly_dark",
        color_discrete_sequence=["#06B6D4"],
    )
    fig_time.update_layout(**_chart_layout())
    st.plotly_chart(fig_time, use_container_width=True)


def render_predict_tab():
    """Live prediction using the best trained model."""
    st.markdown("#### 🔮 Live Prediction with Best Model")

    best_model    = st.session_state.best_model
    best_name     = st.session_state.best_name
    task_type     = st.session_state.task_type
    feature_names = st.session_state.feature_names
    X_test        = st.session_state.X_test

    if best_model is None or feature_names is None:
        st.warning("Run the pipeline first to enable prediction.")
        return

    task_emoji = "🟢 Classification" if task_type == "classification" else "🟠 Regression"
    st.markdown(
        f"<p style='color:#94A3B8;'>Model: <strong style='color:#6C63FF;'>{best_name}</strong>"
        f"&nbsp;|&nbsp; Task: <strong>{task_emoji}</strong></p>",
        unsafe_allow_html=True,
    )

    with st.form("predict_form"):
        st.markdown("**Enter feature values below:**")
        n_cols = min(3, len(feature_names))
        cols = st.columns(n_cols)
        input_vals = {}
        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                mean_val = float(X_test[feat].mean())
                input_vals[feat] = st.number_input(
                    feat, value=round(mean_val, 4), format="%.4f", key=f"pred_{feat}"
                )
        predict_btn = st.form_submit_button("🚀 Predict", type="primary", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([input_vals])
        pred = best_model.predict(input_df)
        result = pred[0]

        if task_type == "classification":
            st.success(f"### 🎯 Predicted Class: `{result}`")
            if hasattr(best_model, "predict_proba"):
                proba  = best_model.predict_proba(input_df)[0]
                proba_df = pd.DataFrame({
                    "Class": [str(c) for c in best_model.classes_],
                    "Probability": proba,
                }).sort_values("Probability", ascending=False)
                fig_prob = px.bar(
                    proba_df, x="Class", y="Probability", color="Probability",
                    color_continuous_scale=["#3B82F6", "#6C63FF", "#22C55E"],
                    title="Class Probabilities", template="plotly_dark",
                )
                fig_prob.update_layout(**_chart_layout(), showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.success(f"### 📈 Predicted Value: `{result:.4f}`")


def render_export_tab():
    """Export comparison CSV and best model pickle."""
    st.markdown("#### 📥 Download Results")

    comparison = st.session_state.comparison
    best_name  = st.session_state.best_name
    task_type  = st.session_state.task_type

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Download Comparison (CSV)",
            comparison.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        pkl_buf = io.BytesIO()
        pickle.dump(st.session_state.best_model, pkl_buf)
        pkl_buf.seek(0)
        st.download_button(
            "🤖 Download Best Model (.pkl)",
            pkl_buf,
            file_name=f"best_model_{best_name.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    st.info(
        f"**Best Model:** {best_name}  \n"
        f"**Task Type:** {task_type}  \n"
        f"Load with: `model = pickle.load(open('model.pkl', 'rb'))`"
    )


# ═══════════════════════════════════════════════════════════════════
# App Header
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    """
<div class="main-header">
    <h1>⚡ Data Science Autometer</h1>
    <p>Automated ML Pipeline &mdash; Upload · Configure · Train · Compare · Export</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Session State ─────────────────────────────────────────────────
for _key in [
    "df", "results", "comparison", "best_name", "best_model",
    "task_type", "models_results", "X_test", "y_test",
    "feature_names", "scaler_obj", "le_dict",
]:
    if _key not in st.session_state:
        st.session_state[_key] = None

# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")
    upload_tab, sample_tab = st.tabs(["Upload", "Sample"])

    with upload_tab:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.results = None

    with sample_tab:
        sample = st.selectbox(
            "Choose sample",
            ["— Select —", "Iris (Classification)", "Titanic (Classification)", "Housing (Regression)"],
        )
        if sample != "— Select —":
            if st.button("Load Sample", use_container_width=True):
                file_map = {
                    "Iris":    "sample_data/iris.csv",
                    "Titanic": "sample_data/titanic.csv",
                    "Housing": "sample_data/housing.csv",
                }
                key = next(k for k in file_map if k in sample)
                st.session_state.df = pd.read_csv(file_map[key])
                st.session_state.results = None
                st.rerun()

    st.divider()

    run_btn = False
    if st.session_state.df is not None:
        df_sb = st.session_state.df
        st.markdown("### ⚙️ Configuration")

        target_col = st.selectbox("🎯 Target Column", df_sb.columns)

        st.markdown("#### 🧠 Task Type")
        task_choice = st.radio(
            "Select task type",
            options=["🤖 Auto-detect", "🟢 Classification", "🟠 Regression"],
            index=0,
            help=(
                "Auto-detect infers task from the target column.\n"
                "Choose Classification for category/label prediction.\n"
                "Choose Regression for continuous numeric prediction."
            ),
        )

        missing_strategy = st.selectbox("🔧 Missing Values", ["mean", "median", "mode", "drop"])
        scale_method     = st.selectbox("📏 Scaling",        ["standard", "minmax", "none"])
        test_size        = st.slider("📊 Test Split", 0.1, 0.5, 0.2, 0.05)
        st.divider()
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ─── Empty State ───────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown(
        """
    <div style="text-align:center;padding:4rem 1rem;">
        <p style="font-size:3.5rem;margin-bottom:0.5rem;">📊</p>
        <h3 style="color:#94A3B8;">Upload a dataset or select a sample to begin</h3>
        <p style="color:#64748B;">Supports CSV and Excel files</p>
        <hr style="border:none;border-top:1px solid #1E293B;margin:2rem auto;width:40%;">
        <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-top:1rem;">
            <div style="background:#1E293B;border-radius:12px;padding:1rem 1.5rem;color:#94A3B8;font-size:0.9rem;">
                🟢 <strong style="color:#22C55E;">Classification</strong><br>Iris · Titanic
            </div>
            <div style="background:#1E293B;border-radius:12px;padding:1rem 1.5rem;color:#94A3B8;font-size:0.9rem;">
                🟠 <strong style="color:#F97316;">Regression</strong><br>Housing Prices
            </div>
            <div style="background:#1E293B;border-radius:12px;padding:1rem 1.5rem;color:#94A3B8;font-size:0.9rem;">
                🤖 <strong style="color:#3B82F6;">Auto-detect</strong><br>Any dataset
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

df   = st.session_state.df
prof = profile_data(df)

# ─── Run Pipeline ──────────────────────────────────────────────────
if run_btn:
    status = st.status("⏳ Running pipeline...", expanded=True)

    status.write("🔧 Handling missing values...")
    df_clean = handle_missing(df, strategy=missing_strategy)

    status.write("🏷️ Encoding categorical features...")
    df_enc, le_dict = encode_categoricals(df_clean, target_col)

    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]
    feature_names = list(X.columns)

    # ── Determine task type ──────────────────────────────────────
    if task_choice == "🟢 Classification":
        task_type = "classification"
        status.write("🟢 Task type set to: **Classification**")
    elif task_choice == "🟠 Regression":
        task_type = "regression"
        status.write("🟠 Task type set to: **Regression**")
    else:
        task_type = detect_task_type(y)
        status.write(f"🔍 Auto-detected task type: **{task_type}**")

    status.write(f"📏 Scaling features ({scale_method})...")
    X_scaled, scaler_obj = scale_features(X, method=scale_method)

    status.write(f"✂️ Splitting data ({int((1 - test_size) * 100)}/{int(test_size * 100)})...")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size)

    status.write("🤖 Training models...")
    models = get_classifiers() if task_type == "classification" else get_regressors()
    models_results = train_models(models, X_train, y_train)

    status.write("📊 Evaluating models...")
    if task_type == "classification":
        comparison = evaluate_classification(models_results, X_test, y_test)
    else:
        comparison = evaluate_regression(models_results, X_test, y_test)
    best_name, best_model = get_best_model(comparison, models_results)

    # Store in session state
    st.session_state.results       = True
    st.session_state.comparison    = comparison
    st.session_state.best_name     = best_name
    st.session_state.best_model    = best_model
    st.session_state.task_type     = task_type
    st.session_state.models_results = models_results
    st.session_state.X_test        = X_test
    st.session_state.y_test        = y_test
    st.session_state.feature_names = feature_names
    st.session_state.scaler_obj    = scaler_obj
    st.session_state.le_dict       = le_dict

    status.update(label="✅ Pipeline complete!", state="complete")

# ─── Tabs ──────────────────────────────────────────────────────────
if st.session_state.results:
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "🏆 Results", "🔮 Predict", "📥 Export"])
else:
    tab1, tab2 = st.tabs(["📋 Data Preview", "📊 Data Profile"])
    tab3 = tab4 = None

# ── Data Preview ───────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Rows",    prof["shape"][0]),
        ("Columns", prof["shape"][1]),
        ("Numeric", len(prof["numeric_cols"])),
        ("Missing", sum(prof["missing"].values())),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{val}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=400)

# ── Data Profile ───────────────────────────────────────────────────
if not st.session_state.results:
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Column Types")
            st.dataframe(
                pd.DataFrame({"Column": prof["dtypes"].keys(), "Type": prof["dtypes"].values()}),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("#### Missing Values")
            st.dataframe(
                pd.DataFrame({
                    "Column":  list(prof["missing"].keys()),
                    "Count":   list(prof["missing"].values()),
                    "Percent": [f"{v}%" for v in prof["missing_pct"].values()],
                }),
                use_container_width=True, hide_index=True,
            )
        if prof["numeric_cols"]:
            st.markdown("#### Numeric Summary")
            st.dataframe(df[prof["numeric_cols"]].describe().round(2), use_container_width=True)

# ── Results Tab ────────────────────────────────────────────────────
if st.session_state.results and tab3 is not None:
    with tab2:
        task_type      = st.session_state.task_type
        comparison     = st.session_state.comparison
        best_name      = st.session_state.best_name
        models_results = st.session_state.models_results
        X_test         = st.session_state.X_test
        y_test         = st.session_state.y_test

        # Badge
        badge_cls = "badge-cls" if task_type == "classification" else "badge-reg"
        emoji     = "🟢" if task_type == "classification" else "🟠"
        st.markdown(
            f'<span class="badge {badge_cls}">{emoji} {task_type.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Best model card
        if task_type == "classification":
            best_score, score_label = comparison.iloc[0]["Accuracy"], "Accuracy"
        else:
            best_score, score_label = comparison.iloc[0]["R² Score"], "R² Score"

        st.markdown(
            f"""
        <div class="best-card">
            <div class="title">🏆 Best Model</div>
            <div class="name">{best_name}</div>
            <div style="color:#94A3B8;margin-top:0.3rem;">
                {score_label}: <strong style="color:#22C55E;">{best_score}</strong>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Comparison table with smart highlighting
        st.markdown("#### 📊 Model Comparison")
        if task_type == "classification":
            highlight_cols = [c for c in comparison.columns if c not in ["Model", "Train Time (s)"]]
            styled = comparison.style.highlight_max(subset=highlight_cols, color="#6C63FF33")
        else:
            r2_cols  = [c for c in ["R² Score"] if c in comparison.columns]
            err_cols = [c for c in ["MAE", "RMSE"] if c in comparison.columns]
            styled = comparison.style
            if r2_cols:
                styled = styled.highlight_max(subset=r2_cols, color="#22C55E33")
            if err_cols:
                styled = styled.highlight_min(subset=err_cols, color="#22C55E33")

        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Task-specific charts
        if task_type == "classification":
            render_classification_charts(comparison, best_name, models_results, X_test, y_test)
        else:
            render_regression_charts(comparison, best_name, models_results, X_test, y_test)

    with tab3:
        render_predict_tab()

    with tab4:
        render_export_tab()
