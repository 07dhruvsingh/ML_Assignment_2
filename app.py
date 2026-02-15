"""
Wine Quality Classification - Streamlit Web Application
=========================================================
Interactive ML dashboard for comparing 6 classification models on the Wine Quality dataset.
Features: Dataset upload, model selection, evaluation metrics, confusion matrix, classification report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Wine Quality ML Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Custom CSS Styling
# ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { background-color: #0e1117; }

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(15, 52, 96, 0.4);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #722f37, #c0392b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }

    .header-gradient {
        background: linear-gradient(135deg, #722f37, #c0392b, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .comparison-table {
        border-radius: 12px;
        overflow: hidden;
    }

    .info-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #722f37;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0e1117 100%);
    }

    .stSelectbox > div > div {
        background-color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)


# ========================
# Helper Functions
# ========================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Random Forest': 'random_forest.joblib',
    'XGBoost': 'xgboost.joblib'
}


@st.cache_resource
def load_models():
    """Load all trained models from the model directory."""
    models = {}
    for name, filename in MODEL_FILES.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
    return models


@st.cache_resource
def load_scaler():
    """Load the fitted scaler."""
    filepath = os.path.join(MODEL_DIR, 'scaler.joblib')
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


@st.cache_data
def load_results():
    """Load pre-computed results."""
    filepath = os.path.join(MODEL_DIR, 'results.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_info():
    """Load dataset information."""
    filepath = os.path.join(MODEL_DIR, 'dataset_info.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_names():
    """Load feature names."""
    filepath = os.path.join(MODEL_DIR, 'feature_names.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def evaluate_uploaded_data(model, X, y_true):
    """Evaluate model on uploaded test data."""
    y_pred = model.predict(X)
    y_proba = None

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    else:
        auc = 0.0

    metrics = {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'AUC': round(auc, 4),
        'Precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'F1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4)
    }

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return metrics, cm, report, y_pred, y_proba


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=['Standard', 'Good'],
        yticklabels=['Standard', 'Good'],
        linewidths=2, linecolor='#1a1a2e',
        ax=ax, cbar_kws={'shrink': 0.8}
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    plt.tight_layout()
    return fig


def plot_roc_curve(model, X, y_true, model_name):
    """Plot ROC curve for the model."""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='#c0392b', lw=2.5, label=f'{model_name} (AUC = {auc_val:.4f})')
        ax.plot([0, 1], [0, 1], color='#8892b0', lw=1.5, linestyle='--', alpha=0.7)
        ax.fill_between(fpr, tpr, alpha=0.15, color='#c0392b')
        ax.set_xlabel('False Positive Rate', fontsize=12, color='white')
        ax.set_ylabel('True Positive Rate', fontsize=12, color='white')
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', color='white', pad=15)
        ax.legend(loc='lower right', fontsize=10, facecolor='#1a1a2e', edgecolor='#0f3460', labelcolor='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.grid(True, alpha=0.1)
        plt.tight_layout()
        return fig
    return None


def plot_metrics_comparison(results):
    """Plot a grouped bar chart comparing all models across metrics."""
    model_names = list(results.keys())
    metrics_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

    data = []
    for model in model_names:
        row = [results[model]['metrics'][m] for m in metrics_names]
        data.append(row)

    df_metrics = pd.DataFrame(data, columns=metrics_names, index=model_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(model_names))
    width = 0.12
    colors = ['#722f37', '#c0392b', '#e74c3c', '#f39c12', '#27ae60', '#2980b9']

    for i, metric in enumerate(metrics_names):
        bars = ax.bar(x + i * width, df_metrics[metric], width, label=metric,
                      color=colors[i], edgecolor='white', linewidth=0.5, alpha=0.9)

    ax.set_xlabel('Models', fontsize=13, color='white', fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, color='white', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10, color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=9, loc='lower right', facecolor='#1a1a2e', edgecolor='#0f3460', labelcolor='white')
    ax.set_ylim(0, 1.15)
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.grid(axis='y', alpha=0.1)
    plt.tight_layout()
    return fig


# ========================
# Main Application
# ========================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🍷 Navigation")
        st.markdown("---")

        page = st.radio(
            "Choose a section:",
            ["🏠 Overview", "📊 Model Comparison", "🔬 Individual Model Analysis", "📁 Upload & Predict"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        This app demonstrates **6 ML classification models** trained on the 
        **Wine Quality Dataset (UCI)**.

        **Models:**
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Naive Bayes (Gaussian)
        - Random Forest
        - XGBoost
        
        **Task:** Predict if wine is of 
        Good Quality (≥7) or Standard Quality (<7)
        """)

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#8892b0; font-size:0.8rem;'>"
            "Built by Dhruv | BITS Pilani M.Tech (AIML/DSE)</p>",
            unsafe_allow_html=True
        )

    # Load resources
    models = load_models()
    results = load_results()
    dataset_info = load_dataset_info()
    feature_names = load_feature_names()
    scaler = load_scaler()

    # ---- OVERVIEW PAGE ----
    if page == "🏠 Overview":
        st.markdown('<h1 class="header-gradient">🍷 Wine Quality Classification Dashboard</h1>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Problem Statement:</strong> Predict the quality of wine based on physicochemical properties 
            measured during wine production. This is a <b>binary classification</b> problem where wines rated 
            <b>≥ 7</b> are classified as <b>Good Quality</b> and wines rated <b>< 7</b> as <b>Standard Quality</b>. 
            The dataset combines both red and white wine variants from the Portuguese "Vinho Verde" region.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Dataset Overview")

        if dataset_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dataset_info['total_instances']}</div>
                    <div class="metric-label">Total Instances</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dataset_info['total_features']}</div>
                    <div class="metric-label">Features</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dataset_info['train_size']}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dataset_info['test_size']}</div>
                    <div class="metric-label">Test Samples</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🏷️ Feature Descriptions")
            feature_descriptions = {
                'fixed_acidity': 'Non-volatile acids that contribute to wine taste (tartaric acid) - g/dm³',
                'volatile_acidity': 'Amount of acetic acid; high levels lead to vinegar taste - g/dm³',
                'citric_acid': 'Adds freshness and flavor to wines - g/dm³',
                'residual_sugar': 'Sugar remaining after fermentation; affects sweetness - g/dm³',
                'chlorides': 'Amount of salt in the wine - g/dm³',
                'free_sulfur_dioxide': 'Free form of SO₂; prevents microbial growth - mg/dm³',
                'total_sulfur_dioxide': 'Total SO₂ (free + bound); detectable at >50 ppm - mg/dm³',
                'density': 'Density of wine; related to alcohol and sugar content - g/cm³',
                'pH': 'Acidity level on 0-14 scale (most wines are 3-4)',
                'sulphates': 'Wine additive contributing to SO₂ levels - g/dm³',
                'alcohol': 'Alcohol content of the wine - % by volume',
                'wine_type': 'Type of wine (0 = Red, 1 = White)'
            }

            feat_df = pd.DataFrame({
                'Feature': feature_descriptions.keys(),
                'Description': feature_descriptions.values()
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="info-box">
                <strong>Source:</strong> {dataset_info['source']}<br>
                <strong>URL:</strong> <a href="{dataset_info['url']}" style="color:#c0392b;">{dataset_info['url']}</a><br>
                <strong>Target:</strong> {dataset_info.get('target_description', 'Binary classification')}<br>
                <strong>Classes:</strong> {', '.join(dataset_info['classes'])}
            </div>
            """, unsafe_allow_html=True)

    # ---- MODEL COMPARISON PAGE ----
    elif page == "📊 Model Comparison":
        st.markdown('<h1 class="header-gradient">📊 Model Comparison</h1>', unsafe_allow_html=True)

        if results:
            # Comparison Table
            st.markdown("### 📋 Evaluation Metrics Comparison")
            comparison_data = []
            for model_name, data in results.items():
                row = {'Model': model_name}
                row.update(data['metrics'])
                comparison_data.append(row)

            df_comparison = pd.DataFrame(comparison_data)
            df_comparison = df_comparison.set_index('Model')

            # Style the dataframe
            styled_df = df_comparison.style.background_gradient(
                cmap='Reds', axis=0
            ).format("{:.4f}")

            st.dataframe(styled_df, use_container_width=True)

            # Best model highlight
            best_model_acc = df_comparison['Accuracy'].idxmax()
            best_acc = df_comparison['Accuracy'].max()
            best_model_f1 = df_comparison['F1'].idxmax()
            best_f1 = df_comparison['F1'].max()

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Accuracy:** {best_model_acc} ({best_acc:.4f})")
            with col2:
                st.success(f"🏆 **Best F1 Score:** {best_model_f1} ({best_f1:.4f})")

            # Visualization
            st.markdown("### 📈 Visual Comparison")
            fig = plot_metrics_comparison(results)
            st.pyplot(fig)
            plt.close()

            # Per-metric bar charts
            st.markdown("### 🔍 Metric-wise Analysis")
            metric_choice = st.selectbox("Select metric to visualize:",
                                         ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            model_names = list(results.keys())
            values = [results[m]['metrics'][metric_choice] for m in model_names]
            colors = ['#722f37', '#c0392b', '#e74c3c', '#f39c12', '#27ae60', '#2980b9']

            bars = ax2.barh(model_names, values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

            for bar, val in zip(bars, values):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center', fontsize=11, color='white', fontweight='bold')

            ax2.set_xlabel(metric_choice, fontsize=12, color='white')
            ax2.set_title(f'{metric_choice} by Model', fontsize=14, fontweight='bold', color='white', pad=15)
            ax2.tick_params(colors='white')
            ax2.set_facecolor('#0e1117')
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_xlim(0, max(values) + 0.15)
            ax2.grid(axis='x', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        else:
            st.warning("⚠️ No results found. Please train the models first by running `train_models.py`.")

    # ---- INDIVIDUAL MODEL ANALYSIS PAGE ----
    elif page == "🔬 Individual Model Analysis":
        st.markdown('<h1 class="header-gradient">🔬 Individual Model Analysis</h1>', unsafe_allow_html=True)

        if models and results:
            selected_model = st.selectbox(
                "🤖 Select a Model:",
                list(models.keys()),
                help="Choose a classification model to view detailed analysis"
            )

            if selected_model in results:
                model_data = results[selected_model]

                # Display metrics as cards
                st.markdown("### 📊 Performance Metrics")
                cols = st.columns(6)
                metric_icons = ['🎯', '📈', '🔬', '📡', '⚡', '🔗']
                for i, (metric, value) in enumerate(model_data['metrics'].items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem;">{metric_icons[i]}</div>
                            <div class="metric-value">{value:.4f}</div>
                            <div class="metric-label">{metric}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Confusion Matrix & ROC Curve side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔲 Confusion Matrix")
                    cm = np.array(model_data['confusion_matrix'])
                    fig_cm = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model}")
                    st.pyplot(fig_cm)
                    plt.close()

                with col2:
                    st.markdown("### 📉 ROC Curve")
                    # Load test data for ROC curve
                    test_data_path = os.path.join(MODEL_DIR, 'test_data.csv')
                    if os.path.exists(test_data_path):
                        test_df = pd.read_csv(test_data_path)
                        X_test = test_df.drop('target', axis=1)
                        y_test = test_df['target']
                        fig_roc = plot_roc_curve(models[selected_model], X_test, y_test, selected_model)
                        if fig_roc:
                            st.pyplot(fig_roc)
                            plt.close()
                        else:
                            st.info("ROC curve not available for this model.")
                    else:
                        st.info("Test data not found. Run train_models.py first.")

                # Classification Report
                st.markdown("### 📝 Classification Report")
                report = model_data['classification_report']
                report_df = pd.DataFrame(report).transpose()
                if 'support' in report_df.columns:
                    report_df['support'] = report_df['support'].apply(lambda x: int(x) if pd.notna(x) else x)
                st.dataframe(
                    report_df.style.background_gradient(cmap='Reds', subset=['precision', 'recall', 'f1-score']),
                    use_container_width=True
                )

        else:
            st.warning("⚠️ Models not found. Please run `train_models.py` first.")

    # ---- UPLOAD & PREDICT PAGE ----
    elif page == "📁 Upload & Predict":
        st.markdown('<h1 class="header-gradient">📁 Upload Data & Predict</h1>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            📌 <strong>Instructions:</strong> Upload a CSV file containing test data with the same features 
            as the Wine Quality dataset. The file should include a <code>target</code> column for evaluation.
            <br><br>
            <em>Expected features: fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, wine_type</em>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📤 Upload your CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.markdown("### 📋 Uploaded Data Preview")
                st.dataframe(df_uploaded.head(10), use_container_width=True)
                st.info(f"📊 Shape: {df_uploaded.shape[0]} rows × {df_uploaded.shape[1]} columns")

                if 'target' in df_uploaded.columns:
                    X_uploaded = df_uploaded.drop('target', axis=1)
                    y_uploaded = df_uploaded['target']

                    # Model selection
                    selected_model = st.selectbox(
                        "🤖 Select Model for Prediction:",
                        list(models.keys())
                    )

                    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                        with st.spinner("Running predictions..."):
                            model = models[selected_model]
                            metrics, cm, report, y_pred, y_proba = evaluate_uploaded_data(
                                model, X_uploaded, y_uploaded
                            )

                            # Display metrics
                            st.markdown("### 📊 Evaluation Results")
                            cols = st.columns(6)
                            metric_icons = ['🎯', '📈', '🔬', '📡', '⚡', '🔗']
                            for i, (metric, value) in enumerate(metrics.items()):
                                with cols[i]:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div style="font-size: 1.5rem;">{metric_icons[i]}</div>
                                        <div class="metric-value">{value:.4f}</div>
                                        <div class="metric-label">{metric}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            # Confusion Matrix
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### 🔲 Confusion Matrix")
                                fig_cm = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model}")
                                st.pyplot(fig_cm)
                                plt.close()

                            with col2:
                                st.markdown("### 📝 Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                if 'support' in report_df.columns:
                                    report_df['support'] = report_df['support'].apply(
                                        lambda x: int(x) if pd.notna(x) else x
                                    )
                                st.dataframe(
                                    report_df.style.background_gradient(
                                        cmap='Reds',
                                        subset=['precision', 'recall', 'f1-score']
                                    ),
                                    use_container_width=True
                                )

                            # Predictions table
                            st.markdown("### 📋 Prediction Results")
                            result_df = df_uploaded.copy()
                            result_df['Predicted'] = y_pred
                            result_df['Correct'] = result_df['target'] == result_df['Predicted']
                            st.dataframe(result_df, use_container_width=True)
                else:
                    st.warning("⚠️ The uploaded file must contain a 'target' column for evaluation.")
                    st.markdown("You can still use the models for prediction:")

                    selected_model = st.selectbox(
                        "🤖 Select Model:",
                        list(models.keys())
                    )

                    if st.button("🚀 Predict", type="primary"):
                        model = models[selected_model]
                        predictions = model.predict(df_uploaded)
                        result_df = df_uploaded.copy()
                        result_df['Predicted_Target'] = predictions
                        st.dataframe(result_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

        else:
            st.markdown("---")
            st.markdown("### 💡 Quick Test")
            st.markdown("Don't have a test file? Use the pre-loaded test data:")

            if st.button("📊 Load Sample Test Data", type="secondary", use_container_width=True):
                test_path = os.path.join(MODEL_DIR, 'test_data.csv')
                if os.path.exists(test_path):
                    test_df = pd.read_csv(test_path)
                    st.dataframe(test_df.head(10), use_container_width=True)
                    st.info(f"📊 Test data shape: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
                    st.download_button(
                        "⬇️ Download Test Data CSV",
                        test_df.to_csv(index=False),
                        "test_data.csv",
                        "text/csv"
                    )


if __name__ == '__main__':
    main()
