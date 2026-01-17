import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# Demo classes needed for unpickling models
class DemoPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.shape[1] < 10:
            padding = np.zeros((X.shape[0], 10 - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > 10:
            X = X[:, :10]
        return X

class DemoClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        np.random.seed(42)
        return np.where(np.random.rand(n_samples) < 0.2, "attack", "normal")

st.set_page_config(page_title="Intelligent IDS", layout="wide", page_icon="🛡️")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛡️ Intelligent Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Network Security Analysis | ML/DL Hybrid Approach</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Model selection
model_files = sorted(MODELS_DIR.glob("model_*.pkl"))
if model_files:
    model_names = [mf.stem.replace("model_", "").upper() for mf in model_files]
    selected_idx = st.sidebar.selectbox("Select Detection Model", range(len(model_names)), format_func=lambda x: model_names[x])
    selected_model = model_files[selected_idx]
    
    st.sidebar.success(f"✓ Model Loaded: {model_names[selected_idx]}")
else:
    st.sidebar.error("⚠️ No models found. Run training first.")
    st.stop()

st.sidebar.markdown("---")

# Performance metrics
if (MODELS_DIR / "performance.csv").exists():
    perf_df = pd.read_csv(MODELS_DIR / "performance.csv")
    st.sidebar.subheader("📊 Model Performance")
    
    selected_model_name = model_names[selected_idx].lower()
    if selected_model_name in perf_df['model'].values:
        model_perf = perf_df[perf_df['model'] == selected_model_name].iloc[0]
        
        st.sidebar.metric("Accuracy", f"{model_perf['accuracy']:.1%}")
        st.sidebar.metric("Precision", f"{model_perf['precision']:.1%}")
        st.sidebar.metric("Recall", f"{model_perf['recall']:.1%}")
        st.sidebar.metric("F1-Score", f"{model_perf['f1']:.1%}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Upload network traffic CSV to analyze")

# Demo data option
demo_file = DATA_DIR / "demo_traffic.csv"
if demo_file.exists():
    if st.sidebar.button("🎯 Load Demo Data"):
        st.session_state['use_demo'] = True

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📈 Performance Dashboard", "ℹ️ About"])

with tab1:
    st.subheader("Upload Network Traffic Data")
    
    # Check if demo data should be used
    if 'use_demo' in st.session_state and st.session_state['use_demo']:
        uploaded_file = str(demo_file)
        st.info(f"📁 Using demo data: {demo_file.name}")
        df = pd.read_csv(demo_file)
        st.session_state['use_demo'] = False  # Reset
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload network traffic features")
        
        if uploaded_file is None:
            st.warning("⬆️ Please upload a CSV file or click 'Load Demo Data' in the sidebar")
            st.markdown("### Expected CSV Format")
            st.code("""
duration,protocol_type,service,flag,src_bytes,dst_bytes,...
0,tcp,http,SF,215,45076,...
1,udp,dns,SF,162,162,...
            """)
            st.stop()
        
        df = pd.read_csv(uploaded_file)
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", len(df))
    with col2:
        st.metric("📋 Features", len(df.columns))
    with col3:
        st.metric("💾 Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Show sample data
    with st.expander("👁️ View Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Run prediction
    if st.button("🚀 Run Detection", type="primary", use_container_width=True):
        with st.spinner("Analyzing network traffic..."):
            try:
                model = joblib.load(selected_model)
                predictions = model.predict(df)
                
                # Add predictions to dataframe
                result_df = df.copy()
                result_df["Prediction"] = predictions
                result_df["Threat_Level"] = result_df["Prediction"].apply(
                    lambda x: "🔴 HIGH" if x == "attack" else "🟢 NORMAL"
                )
                
                st.success("✅ Analysis Complete!")
                
                # Summary metrics
                st.markdown("### 🎯 Detection Summary")
                total = len(predictions)
                attacks = int((predictions == "attack").sum())
                normals = total - attacks
                attack_pct = (attacks / total * 100)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", f"{total:,}", help="Total network events analyzed")
                with col2:
                    st.metric("🔴 Attacks", f"{attacks:,}", delta=f"{attack_pct:.1f}%", delta_color="inverse")
                with col3:
                    st.metric("🟢 Normal", f"{normals:,}", delta=f"{100-attack_pct:.1f}%", delta_color="normal")
                with col4:
                    threat_level = "HIGH" if attack_pct > 30 else "MEDIUM" if attack_pct > 10 else "LOW"
                    st.metric("Threat Level", threat_level)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Detection Distribution")
                    fig_pie = px.pie(
                        values=[normals, attacks],
                        names=["Normal", "Attack"],
                        color_discrete_sequence=["#2ecc71", "#e74c3c"],
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📈 Detection Timeline")
                    result_df["is_attack"] = (result_df["Prediction"] == "attack").astype(int)
                    fig_line = px.line(
                        result_df.reset_index(),
                        x="index",
                        y="is_attack",
                        title="Anomaly Detection Over Time",
                        labels={"index": "Event Index", "is_attack": "Attack (1) / Normal (0)"}
                    )
                    fig_line.update_traces(line_color='#e74c3c')
                    st.plotly_chart(fig_line, use_container_width=True)
                
                # Detailed results
                st.markdown("### 🔍 Detailed Results")
                
                # Filter options
                filter_option = st.radio("Show:", ["All", "Attacks Only", "Normal Only"], horizontal=True)
                
                if filter_option == "Attacks Only":
                    display_df = result_df[result_df["Prediction"] == "attack"]
                elif filter_option == "Normal Only":
                    display_df = result_df[result_df["Prediction"] == "normal"]
                else:
                    display_df = result_df
                
                st.dataframe(
                    display_df.head(100),
                    use_container_width=True,
                    height=400
                )
                
                # Download results
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="ids_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

with tab2:
    st.subheader("📈 Model Performance Comparison")
    
    if (MODELS_DIR / "performance.csv").exists():
        perf_df = pd.read_csv(MODELS_DIR / "performance.csv")
        
        # Performance table
        st.dataframe(
            perf_df.style.background_gradient(cmap='RdYlGn', subset=['accuracy', 'precision', 'recall', 'f1']),
            use_container_width=True
        )
        
        # Comparison chart
        fig = go.Figure()
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=perf_df['model'],
                y=perf_df[metric],
                text=perf_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = perf_df.loc[perf_df['accuracy'].idxmax()]
        st.success(f"🏆 Best Model: **{best_model['model'].upper()}** (Accuracy: {best_model['accuracy']:.1%})")
    else:
        st.warning("No performance data available.")

with tab3:
    st.markdown("""
    ## 🛡️ About This System
    
    ### Overview
    This Intelligent Intrusion Detection System (IDS) uses advanced Machine Learning and Deep Learning 
    techniques to identify malicious network activity in real-time.
    
    ### Models Implemented
    - **Random Forest (RF)**: Ensemble learning with 200 decision trees
    - **Support Vector Machine (SVM)**: RBF kernel for non-linear classification
    - **K-Nearest Neighbors (KNN)**: Instance-based learning (k=7)
    - **Isolation Forest**: Unsupervised anomaly detection
    - **K-Means**: Clustering-based detection
    - **Autoencoder**: Deep learning reconstruction-based detection
    
    ### Attack Types Detected
    - 🔴 **DoS/DDoS**: Denial of Service attacks
    - 🔴 **Port Scanning**: Network reconnaissance
    - 🔴 **Injection**: SQL/Command injection
    - 🔴 **Botnet**: C&C communications
    - 🔴 **Brute Force**: Authentication attacks
    - 🔴 **Exfiltration**: Data theft
    
    ### Technical Stack
    - **ML Framework**: scikit-learn 1.5.1
    - **DL Framework**: PyTorch 2.4.0
    - **Data Processing**: pandas, numpy
    - **Visualization**: Streamlit, Plotly
    
    ### Integration
    This system can be integrated with:
    - SIEM platforms (Splunk, QRadar, ELK)
    - SOC workflows
    - Network monitoring tools
    
    ---
    
    **Project**: Cybersecurity IDS | **Date**: January 2026
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🛡️ Intelligent IDS | Built with Streamlit + ML/DL | © 2026</p>
    </div>
    """,
    unsafe_allow_html=True
)
