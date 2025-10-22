"""
Generative AI-Powered Data Analytics Platform
Role-Based Interface System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
import logging

# Import custom modules
from config import Config
from utils.data_generator import DataGenerator
from utils.session_data import get_active_data
from agents.data_processor import DataProcessor
from agents.ai_core import GenerativeAnalyticsCore
from agents.nlp_interface import NaturalLanguageInterface
from agents.dynamic_dashboard import create_dashboard_tab
from agents.scenario_simulator import render_scenario_simulator
from agents.system_optimizer import render_system_optimizer
from agents.blockchain_manager import render_blockchain_manager
from agents.federated_learning import render_federated_learning
try:
    from agents.advanced_ml_models import render_advanced_ml_models
except Exception as e:
    render_advanced_ml_models = None
    logging.error(f"Failed to import Advanced ML module: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

def role_selection_page():
    """Role selection page for user login"""
    st.set_page_config(
        page_title="AI Analytics Platform - Login",
        page_icon="🤖",
        layout="centered"
    )
    
    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
    }
    .role-card {
        background: rgba(255,255,255,0.15);
        padding: 2.5rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    .role-card:hover {
        border-color: #fff;
        background: rgba(255,255,255,0.25);
        transform: translateY(-8px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .role-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .role-description {
        font-size: 1.1rem;
        opacity: 0.9;
        line-height: 1.8;
    }
    .platform-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .platform-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-container">
        <div class="platform-title">🤖 AI Analytics Platform</div>
        <div class="platform-subtitle">Choose Your Role to Access Specialized Tools</div>
        <p style="font-size: 1.1rem; opacity: 0.8;">Select your role to unlock features designed specifically for your workflow and expertise level</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="role-card">
            <div class="role-title">🔬 Data Scientist / Analyst</div>
            <div class="role-description">
                Advanced technical workspace for data professionals
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Enter Data Scientist Workspace", key="data_scientist", use_container_width=True, type="primary"):
            st.session_state.user_role = "data_scientist"
            st.rerun()
        
        st.markdown("""
        **🎯 Perfect for:**
        - 📥 Automated data ingestion & cleaning
        - 🤖 AI-driven predictive modeling
        - 🔍 Anomaly detection & analysis
        - 💬 Natural language data queries
        - 📊 Dynamic exploration dashboards
        - 🧪 Advanced statistical analysis
        - 🔧 Model optimization & tuning
        """)
    
    with col2:
        st.markdown("""
        <div class="role-card">
            <div class="role-title">📊 Business Decision Maker</div>
            <div class="role-description">
                Strategic insights dashboard for executives and managers
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Enter Executive Dashboard", key="business_user", use_container_width=True, type="primary"):
            st.session_state.user_role = "business_user"
            st.rerun()
        
        st.markdown("""
        **🎯 Perfect for:**
        - 🎯 Contextual scenario-based analytics
        - 🔮 What-if analysis & forecasting
        - 🛡️ Privacy-secure federated learning
        - 📋 Explainable AI summaries
        - ⚡ Real-time alerts & notifications
        - 📈 Trend analysis & insights
        - 🎪 Executive reporting dashboards
        """)

def data_scientist_interface():
    """Specialized interface for Data Scientists and Analysts"""
    st.set_page_config(
        page_title="Data Scientist Workspace",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components with error handling
    try:
        data_generator = DataGenerator()
        data_processor = DataProcessor()
        ai_core = GenerativeAnalyticsCore()
        nlp_interface = NaturalLanguageInterface()
        
    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        logger.error(f"Component initialization error: {str(e)}")
        st.stop()
    
    # Header with role indicator
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🔄 Switch Role", key="switch_role_ds"):
            del st.session_state.user_role
            st.rerun()
    with col2:
        st.markdown("# 🔬 Data Scientist Workspace")
        st.markdown("*Advanced Analytics • Machine Learning • Deep Insights*")
    with col3:
        st.info("👤 Data Scientist")
    
    # Sidebar for quick actions and workspace stats
    with st.sidebar:
        st.markdown("## 🚀 Quick Actions")
        
        # Data ingestion shortcut
        if st.button("📥 Quick Data Upload", use_container_width=True):
            st.session_state.active_tab = "data_ingestion"
        
        # Model training shortcut
        if st.button("🤖 Train ML Models", use_container_width=True):
            st.session_state.active_tab = "ml_modeling"
        
        # Analysis shortcut
        if st.button("📊 Run Analysis", use_container_width=True):
            st.session_state.active_tab = "advanced_analytics"
        
        # NLP Query shortcut
        if st.button("💬 Ask Data Question", use_container_width=True):
            st.session_state.active_tab = "nl_queries"
        
        st.markdown("---")
        st.markdown("## 📈 Workspace Stats")
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            st.metric("Dataset Size", f"{len(st.session_state.processed_data):,} rows")
            st.metric("Features", f"{len(st.session_state.processed_data.columns)} columns")
            st.metric("Memory Usage", f"{st.session_state.processed_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            st.info("No data loaded yet")
        
        st.markdown("---")
        st.markdown("## ⚙️ System Status")
        st.success("✅ All systems operational")
        st.info(f"🔢 Max rows: {config.MAX_ROWS:,}")
        st.info(f"📊 Max file: {config.MAX_FILE_SIZE_MB}MB")
    
    # Main tabs for data scientist workflow
    tabs = st.tabs([
        "📥 Data Ingestion & Cleaning",
        "🔍 Exploratory Analysis", 
        "🤖 ML Modeling & Evaluation",
        "💬 Natural Language Queries",
        "⚡ Advanced Analytics",
        "🔧 System Optimization",
        "📋 Processing Logs"
    ])
    
    with tabs[0]:
        data_ingestion_cleaning_tab(data_processor, data_generator)
    
    with tabs[1]:
        exploratory_analysis_tab(data_processor, ai_core)
    
    with tabs[2]:
        ml_modeling_tab()
    
    with tabs[3]:
        nl_queries_tab(nlp_interface)
    
    with tabs[4]:
        advanced_analytics_tab(ai_core, data_processor)
    
    with tabs[5]:
        system_optimization_tab()
    
    with tabs[6]:
        processing_logs_tab()

def business_user_interface():
    """Specialized interface for Business Decision Makers"""
    st.set_page_config(
        page_title="Executive Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components with error handling
    try:
        ai_core = GenerativeAnalyticsCore()
        data_processor = DataProcessor()
        data_generator = DataGenerator()
        
    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        logger.error(f"Component initialization error: {str(e)}")
        st.stop()
    
    # Header with role indicator
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🔄 Switch Role", key="switch_role_bu"):
            del st.session_state.user_role
            st.rerun()
    with col2:
        st.markdown("# 📊 Executive Dashboard")
        st.markdown("*Strategic Insights • Decision Support • Business Intelligence*")
    with col3:
        st.info("👤 Executive")
    
    # Sidebar for executive summary
    with st.sidebar:
        st.markdown("## 📈 Executive Summary")
        
        # Key performance indicators
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Quality", "94%", "↗️ 2%")
            st.metric("Forecast Accuracy", "87%", "↗️ 5%")
        with col2:
            st.metric("Insights Generated", "47", "↗️ 12")
            st.metric("Risk Score", "Low", "↘️ 15%")
        
        st.markdown("---")
        st.markdown("## 🎯 Quick Insights")
        st.success("✅ Revenue trending upward (+12%)")
        st.warning("⚠️ Customer churn rate increasing")
        st.info("ℹ️ New market opportunity detected")
        st.error("🚨 Inventory levels critically low")
        
        st.markdown("---")
        st.markdown("## 🚀 Quick Actions")
        if st.button("📧 Generate Executive Report", use_container_width=True):
            st.success("📧 Executive report generated and sent!")
        
        if st.button("📱 Send Alert to Team", use_container_width=True):
            st.success("📱 Alert sent to decision makers!")
        
        if st.button("💾 Export Dashboard", use_container_width=True):
            st.success("💾 Dashboard exported successfully!")
    
    # Main tabs for business user workflow
    tabs = st.tabs([
        "📥 Data Ingestion & Cleaning",
        "📊 Executive Dashboard",
        "🎯 Scenario Planning",
        "🔮 Predictive Insights",
        "🛡️ Privacy & Security",
        "📈 Trend Analysis",
        "⚡ Real-time Alerts"
    ])
    
    with tabs[0]:
        data_ingestion_cleaning_tab(data_processor, data_generator)

    with tabs[1]:
        executive_dashboard_tab(ai_core)
    
    with tabs[2]:
        scenario_planning_tab()
    
    with tabs[3]:
        predictive_insights_tab(ai_core)
    
    with tabs[4]:
        privacy_security_tab()
    
    with tabs[5]:
        trend_analysis_tab(ai_core)
    
    with tabs[6]:
        realtime_alerts_tab()

# Data Scientist Tab Functions
def data_ingestion_cleaning_tab(data_processor, data_generator):
    """Data ingestion and cleaning tab for data scientists"""
    st.header("📥 Automated Data Ingestion & Cleaning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help=f"Maximum file size: {config.MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            try:
                # Read and process the data
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.session_state.processed_data = data
                st.session_state.uploaded_data = data
                st.session_state.data_source = 'upload'
                
                st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                
                # Display basic info
                st.subheader("📊 Dataset Overview")
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    st.metric("Rows", f"{len(data):,}")
                with col1b:
                    st.metric("Columns", len(data.columns))
                with col1c:
                    st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                # Show data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Automated cleaning options
                st.subheader("🧹 Automated Cleaning Options")
                
                col2a, col2b = st.columns(2)
                with col2a:
                    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
                    handle_missing = st.selectbox("Handle missing values", 
                                                ["Drop rows", "Fill with mean", "Fill with median", "Forward fill"])
                with col2b:
                    remove_outliers = st.checkbox("Remove statistical outliers")
                    standardize_columns = st.checkbox("Standardize column names")
                
                if st.button("🚀 Apply Automated Cleaning", type="primary"):
                    with st.spinner("Cleaning data..."):
                        cleaned_data = data.copy()
                        
                        if remove_duplicates:
                            cleaned_data = cleaned_data.drop_duplicates()
                            st.info(f"Removed {len(data) - len(cleaned_data)} duplicate rows")
                        
                        if handle_missing == "Drop rows":
                            cleaned_data = cleaned_data.dropna()
                        elif handle_missing == "Fill with mean":
                            cleaned_data = cleaned_data.fillna(cleaned_data.mean(numeric_only=True))
                        elif handle_missing == "Fill with median":
                            cleaned_data = cleaned_data.fillna(cleaned_data.median(numeric_only=True))
                        elif handle_missing == "Forward fill":
                            cleaned_data = cleaned_data.fillna(method='ffill')
                        
                        if standardize_columns:
                            cleaned_data.columns = [col.lower().replace(' ', '_') for col in cleaned_data.columns]
                        
                        st.session_state.processed_data = cleaned_data
                        st.session_state.uploaded_data = cleaned_data
                        st.success("✅ Data cleaning completed!")
                        
                        # Show cleaning results
                        col3a, col3b = st.columns(2)
                        with col3a:
                            st.metric("Original Rows", f"{len(data):,}")
                            st.metric("Original Columns", len(data.columns))
                        with col3b:
                            st.metric("Cleaned Rows", f"{len(cleaned_data):,}")
                            st.metric("Cleaned Columns", len(cleaned_data.columns))
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.subheader("🔬 Generate Synthetic Data")
        st.markdown("*For testing and development*")
        
        data_type = st.selectbox("Data Type", 
                                ["Sales Data", "Customer Data", "Financial Data", "IoT Sensor Data"])
        num_rows = st.slider("Number of Rows", 100, 10000, 1000)
        
        if st.button("Generate Data", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                try:
                    synthetic_data = data_generator.generate_sample_data(data_type.lower().replace(' ', '_'), num_rows)
                    st.session_state.data = synthetic_data
                    st.session_state.processed_data = synthetic_data
                    st.session_state.uploaded_data = synthetic_data
                    st.success(f"✅ Generated {len(synthetic_data)} rows of {data_type}")
                    preview_rows = st.slider('Preview rows', 10, min(1000, len(synthetic_data)), min(100, len(synthetic_data)), key='synthetic_preview_rows')
                    st.dataframe(synthetic_data.head(preview_rows), use_container_width=True)
                    st.download_button(
                        label='Download CSV',
                        data=synthetic_data.to_csv(index=False),
                        file_name=f"{data_type.lower().replace(' ', '_')}_{len(synthetic_data)}.csv",
                        mime='text/csv'
                    )
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")

def exploratory_analysis_tab(data_processor, ai_core):
    """Exploratory data analysis tab"""
    st.header("🔍 Exploratory Data Analysis")
    
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first in the Data Ingestion tab.")
        return
    
    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution Analysis
    st.subheader("📈 Distribution Analysis")
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for distribution", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(data, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

def ml_modeling_tab():
    """Machine learning modeling tab"""
    st.header("🤖 ML Modeling & Evaluation")
    
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first.")
        return
    
    st.subheader("🎯 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type", 
                                 ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"])
        target_column = st.selectbox("Target Column", data.columns)
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
        cross_validation = st.checkbox("Use Cross Validation", value=True)
    
    feature_columns = st.multiselect("Feature Columns", 
                                   [col for col in data.columns if col != target_column],
                                   default=[col for col in data.columns if col != target_column][:5])
    
    if st.button("🚀 Train Model", type="primary"):
        if feature_columns:
            with st.spinner("Training model..."):
                try:
                    # Simulate model training
                    import time
                    time.sleep(2)  # Simulate training time
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display mock results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", "87.3%")
                    with col2:
                        st.metric("R² Score", "0.85")
                    with col3:
                        st.metric("RMSE", "0.23")
                    
                    # Feature importance (mock)
                    st.subheader("📊 Feature Importance")
                    importance_data = pd.DataFrame({
                        'Feature': feature_columns[:5],
                        'Importance': np.random.rand(len(feature_columns[:5]))
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(importance_data, x='Importance', y='Feature', 
                               orientation='h', title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
        else:
            st.error("Please select at least one feature column.")

def nl_queries_tab(nlp_interface):
    """Natural language queries tab in chat-style with session history"""
    st.header("💬 Natural Language Data Queries")

    # Resolve data
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first.")
        return

    # Initialize chat sessions state
    if 'nl_chat_sessions' not in st.session_state:
        st.session_state['nl_chat_sessions'] = {"Chat 1": []}
    if 'nl_active_chat' not in st.session_state:
        st.session_state['nl_active_chat'] = "Chat 1"
    if 'nl_chat_counter' not in st.session_state:
        st.session_state['nl_chat_counter'] = 1

    st.markdown("*Ask questions about your data in plain English*")

    # Layout: sidebar-like column for chat sessions + main chat area
    left, right = st.columns([1, 3])

    with left:
        st.subheader("💬 Chats")
        sessions = st.session_state['nl_chat_sessions']
        chat_names = list(sessions.keys())
        # Select active chat
        active_idx = chat_names.index(st.session_state['nl_active_chat']) if st.session_state['nl_active_chat'] in chat_names else 0
        selected_chat = st.selectbox("Select chat", chat_names, index=active_idx)
        if selected_chat != st.session_state['nl_active_chat']:
            st.session_state['nl_active_chat'] = selected_chat

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state['nl_chat_counter'] += 1
                new_name = f"Chat {st.session_state['nl_chat_counter']}"
                sessions[new_name] = []
                st.session_state['nl_active_chat'] = new_name
        with col_b:
            if st.button("🗑️ Delete Chat", use_container_width=True):
                # Do not allow deleting the last remaining chat
                if len(sessions) > 1:
                    current = st.session_state['nl_active_chat']
                    del sessions[current]
                    st.session_state['nl_active_chat'] = list(sessions.keys())[0]

        if st.button("🧹 Clear Conversation", use_container_width=True):
            sessions[st.session_state['nl_active_chat']] = []
            # Also clear model-side history
            try:
                nlp_interface.clear_conversation_history()
            except Exception:
                pass

        # Helpful suggestions based on dataset
        st.markdown("---")
        st.caption("Suggestions")
        suggestions = []
        try:
            suggestions = nlp_interface.get_query_suggestions(data)
        except Exception:
            suggestions = [
                "Show me a summary of this data",
                "What are the main statistics?",
            ]
        # Render suggestion buttons
        queued_prompt = None
        for s in suggestions:
            if st.button(s):
                queued_prompt = s

    with right:
        st.subheader("Conversation")
        messages = st.session_state['nl_chat_sessions'][st.session_state['nl_active_chat']]

        # Render existing messages
        for i, msg in enumerate(messages):
            role = msg.get('role', 'assistant')
            with st.chat_message(role):
                st.write(msg.get('content', ''))
                # Render visual outputs for assistant messages
                if role == 'assistant':
                    res = msg.get('result', {})
                    intent = res.get('intent', 'general')
                    if intent == 'statistics':
                        stats = res.get('statistics', {})
                        if stats:
                            stats_df = pd.DataFrame(stats).T
                            st.dataframe(stats_df, use_container_width=True)
                            if 'mean' in stats_df.columns:
                                fig = px.bar(stats_df.reset_index(), x='index', y='mean', title='Mean values by column')
                                st.plotly_chart(fig, use_container_width=True, key=f"nlq_stats_mean_{st.session_state['nl_active_chat']}_{i}")
                    elif intent == 'aggregation':
                        aggs = res.get('aggregations', {})
                        if aggs:
                            first_key = next(iter(aggs))
                            agg_map = aggs[first_key]
                            agg_df = pd.DataFrame(list(agg_map.items()), columns=['group', 'value'])
                            fig = px.bar(agg_df, x='group', y='value', title=first_key.replace('_', ' ').title())
                            st.plotly_chart(fig, use_container_width=True, key=f"nlq_agg_{st.session_state['nl_active_chat']}_{i}_{first_key}")
                    elif intent == 'comparison':
                        comps = res.get('comparisons', {})
                        if comps:
                            comp_name = next(iter(comps))
                            parts = comp_name.split('_vs_')
                            if len(parts) == 2 and parts[0] in data.columns and parts[1] in data.columns:
                                fig = px.scatter(data, x=parts[0], y=parts[1], title=f"{parts[0]} vs {parts[1]}")
                                st.plotly_chart(fig, use_container_width=True, key=f"nlq_comp_{st.session_state['nl_active_chat']}_{i}_{parts[0]}_{parts[1]}")
                            corr = comps[comp_name].get('correlation')
                            if corr is not None:
                                st.info(f"Correlation: {corr:.3f}")
                    elif intent == 'visualization':
                        vizzes = res.get('visualizations', [])
                        if vizzes:
                            viz = vizzes[0]
                            if viz.get('type') == 'scatter' and viz.get('x') and viz.get('y'):
                                fig = px.scatter(data, x=viz['x'], y=viz['y'], title=viz.get('description', 'Scatter plot'))
                                st.plotly_chart(fig, use_container_width=True, key=f"nlq_viz_scatter_{st.session_state['nl_active_chat']}_{i}_{viz.get('x')}_{viz.get('y')}")
                            elif viz.get('type') == 'bar' and viz.get('x'):
                                counts = data[viz['x']].value_counts(dropna=False).reset_index()
                                counts.columns = [viz['x'], 'count']
                                fig = px.bar(counts, x=viz['x'], y='count', title=viz.get('description', 'Bar chart'))
                                st.plotly_chart(fig, use_container_width=True, key=f"nlq_viz_bar_{st.session_state['nl_active_chat']}_{i}_{viz.get('x')}")
                            elif viz.get('type') == 'histogram' and viz.get('x'):
                                fig = px.histogram(data, x=viz['x'], nbins=30, title=viz.get('description', 'Histogram'))
                                st.plotly_chart(fig, use_container_width=True, key=f"nlq_viz_hist_{st.session_state['nl_active_chat']}_{i}_{viz.get('x')}")
                    else:
                        sample = res.get('data')
                        if sample:
                            st.dataframe(pd.DataFrame(sample), use_container_width=True)

        # Chat input
        user_input = st.chat_input("Ask a question about your data")
        if user_input or queued_prompt:
            q = user_input if user_input else queued_prompt
            # Append user message
            messages.append({'role': 'user', 'content': q})
            with st.spinner("Processing your question..."):
                try:
                    result_dict = nlp_interface.process_query(q, data)
                    if not result_dict.get('success', False):
                        # Store error as assistant message
                        messages.append({'role': 'assistant', 'content': result_dict.get('response', 'Failed to process query.'), 'result': {}})
                    else:
                        messages.append({
                            'role': 'assistant',
                            'content': result_dict.get('response', ''),
                            'result': result_dict.get('result', {})
                        })
                        st.success("✅ Query processed successfully!")
                except Exception as e:
                    messages.append({'role': 'assistant', 'content': f"❌ Error processing query: {str(e)}", 'result': {}})
            # Persist back
            st.session_state['nl_chat_sessions'][st.session_state['nl_active_chat']] = messages

def advanced_analytics_tab(ai_core, data_processor):
    """Advanced analytics tab"""
    st.header("⚡ Advanced Analytics")
    
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first.")
        return
    
    # Analytics options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Anomaly Detection")
        if st.button("Detect Anomalies", use_container_width=True):
            with st.spinner("Detecting anomalies..."):
                # Mock anomaly detection
                anomalies_found = np.random.randint(1, 10)
                st.success(f"✅ Found {anomalies_found} potential anomalies")
                
                # Mock anomaly visualization
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.scatter(data, x=range(len(data)), y=col,
                                   title=f"Anomaly Detection in {col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Trend Analysis")
        if st.button("Analyze Trends", use_container_width=True):
            with st.spinner("Analyzing trends..."):
                # Mock trend analysis
                st.success("✅ Trend analysis completed")
                
                # Mock trend visualization
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    # Generate a safe time index for large datasets to avoid pandas bounds overflow
                    n = len(data)
                    if n <= 365:
                        freq = 'D'
                    elif n <= 24 * 365:
                        freq = 'H'
                    elif n <= 60 * 24 * 365:
                        freq = 'min'
                    else:
                        freq = 'S'
                    trend_data = pd.DataFrame({
                        'Time': pd.date_range('2023-01-01', periods=n, freq=freq),
                        'Value': data[col].values
                    })
                    fig = px.line(trend_data, x='Time', y='Value',
                                title=f"Trend Analysis for {col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Analytics
    st.subheader("🔮 Predictive Analytics")
    
    col3, col4 = st.columns(2)
    with col3:
        forecast_periods = st.slider("Forecast Periods", 1, 30, 7)
    with col4:
        confidence_interval = st.slider("Confidence Interval", 80, 99, 95)
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Mock forecasting
            st.success("✅ Forecast generated successfully!")
            
            # Mock forecast visualization
            dates = pd.date_range('2024-01-01', periods=forecast_periods)
            forecast_values = np.random.randn(forecast_periods).cumsum() + 100
            
            forecast_df = pd.DataFrame({
                'Date': dates,
                'Forecast': forecast_values,
                'Lower_Bound': forecast_values - 5,
                'Upper_Bound': forecast_values + 5
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'],
                                   mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper_Bound'],
                                   fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower_Bound'],
                                   fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                                   name=f'{confidence_interval}% Confidence Interval'))
            
            fig.update_layout(title="Predictive Forecast", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

def system_optimization_tab():
    """System optimization tab"""
    st.header("🔧 System Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Performance Metrics")
        st.metric("CPU Usage", "45%", "-5%")
        st.metric("Memory Usage", "2.3 GB", "+0.2 GB")
        st.metric("Processing Speed", "1.2k rows/sec", "+200")
    
    with col2:
        st.subheader("🔧 Optimization Options")
        enable_caching = st.checkbox("Enable Data Caching", value=True)
        parallel_processing = st.checkbox("Parallel Processing", value=True)
        memory_optimization = st.checkbox("Memory Optimization", value=False)
        
        if st.button("Apply Optimizations", type="primary"):
            st.success("✅ Optimizations applied successfully!")

def processing_logs_tab():
    """Processing logs tab"""
    st.header("📋 Processing Logs")
    
    # Initialize logs if not exists
    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = [
            {"timestamp": "2024-01-15 10:30:00", "level": "INFO", "message": "System initialized successfully"},
            {"timestamp": "2024-01-15 10:31:15", "level": "INFO", "message": "Data uploaded: 1000 rows, 5 columns"},
            {"timestamp": "2024-01-15 10:32:30", "level": "WARNING", "message": "Missing values detected in column 'age'"},
            {"timestamp": "2024-01-15 10:33:45", "level": "INFO", "message": "Data cleaning completed"},
            {"timestamp": "2024-01-15 10:35:00", "level": "SUCCESS", "message": "Model training completed with 87% accuracy"}
        ]
    
    # Display logs
    for log in reversed(st.session_state.processing_logs[-20:]):  # Show last 20 logs
        if log["level"] == "ERROR":
            st.error(f"🔴 {log['timestamp']} - {log['message']}")
        elif log["level"] == "WARNING":
            st.warning(f"🟡 {log['timestamp']} - {log['message']}")
        elif log["level"] == "SUCCESS":
            st.success(f"🟢 {log['timestamp']} - {log['message']}")
        else:
            st.info(f"🔵 {log['timestamp']} - {log['message']}")
    
    if st.button("Clear Logs"):
        st.session_state.processing_logs = []
        st.success("Logs cleared!")

# Business User Tab Functions
def executive_dashboard_tab(ai_core):
    """Executive dashboard driven by the uploaded dataset"""
    st.header("📊 Executive Dashboard")

    # Use uploaded/processed data only
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first in the Data Ingestion tab.")
        return

    # KPIs derived from dataset
    st.subheader("📈 Key Performance Indicators")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    kpi_col = numeric_cols[0] if len(numeric_cols) > 0 else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(data):,}")
        st.metric("Columns", f"{len(data.columns)}")

    with col2:
        total_cells = len(data) * len(data.columns)
        missing_cells = int(data.isnull().sum().sum())
        data_quality = 0 if total_cells == 0 else (1 - missing_cells / total_cells) * 100
        mem_mb = data.memory_usage(deep=True).sum() / 1024**2
        st.metric("Data Quality", f"{data_quality:.0f}%")
        st.metric("Memory Usage", f"{mem_mb:.1f} MB")

    with col3:
        if kpi_col:
            st.metric(f"Avg {kpi_col}", f"{data[kpi_col].mean():.2f}")
            st.metric(f"Max {kpi_col}", f"{data[kpi_col].max():.2f}")
        else:
            st.metric("Numeric Columns", "None")
            st.metric("Max (n/a)", "-")

    with col4:
        if categorical_cols:
            cat = categorical_cols[0]
            st.metric(f"Unique {cat}", f"{data[cat].nunique():,}")
            top_cat = data[cat].value_counts(dropna=False).idxmax()
            st.metric(f"Top {cat}", str(top_cat))
        else:
            st.metric("Categoricals", "None")
            st.metric("Top (n/a)", "-")

    # Charts from dataset
    st.subheader("📊 Business Intelligence Charts")

    colA, colB = st.columns(2)

    # Detect a time column
    time_col = None
    for c in data.columns:
        if np.issubdtype(data[c].dtype, np.datetime64):
            time_col = c
            break
    if time_col is None:
        for c in data.columns:
            if 'date' in str(c).lower():
                try:
                    data[c] = pd.to_datetime(data[c], errors='coerce')
                    if data[c].notna().any():
                        time_col = c
                        break
                except Exception:
                    pass

    with colA:
        if time_col and kpi_col:
            tmp = data[[time_col, kpi_col]].dropna().copy()
            tmp['period'] = tmp[time_col].dt.to_period('M')
            trend = tmp.groupby('period')[kpi_col].sum().reset_index()
            trend['period'] = trend['period'].astype(str)
            fig = px.line(trend, x='period', y=kpi_col, title=f"{kpi_col} Trend by Month")
            st.plotly_chart(fig, use_container_width=True)
        elif kpi_col:
            fig = px.histogram(data, x=kpi_col, title=f"Distribution of {kpi_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric column available for trend or distribution chart.")

    with colB:
        if categorical_cols:
            cat = categorical_cols[0]
            counts = data[cat].value_counts(dropna=False).reset_index()
            counts.columns = [cat, 'count']
            fig = px.pie(counts, names=cat, values='count', title=f"Distribution of {cat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical column available for segment distribution.")

    # Data-driven insights
    st.subheader("🧠 AI-Generated Strategic Insights")
    insights = []
    if kpi_col:
        mean_v = data[kpi_col].mean()
        median_v = data[kpi_col].median()
        max_v = data[kpi_col].max()
        insights.append(f"Average {kpi_col} is {mean_v:.2f}; median {median_v:.2f}.")
        insights.append(f"Peak {kpi_col} observed at {max_v:.2f}.")
    if categorical_cols:
        cat = categorical_cols[0]
        top = data[cat].value_counts(dropna=False).head(1)
        if not top.empty:
            insights.append(f"Top {cat} is '{top.index[0]}' with {int(top.iloc[0])} records.")
    if time_col and kpi_col:
        insights.append(f"{kpi_col} shows monthly variability; see trend chart above.")
    if not insights:
        insights = ["Dataset loaded. Map domain-specific fields to tailor KPIs."]

    for insight in insights:
        st.info(insight)

def scenario_planning_tab():
    """Scenario planning and what-if analysis (requires uploaded data)"""
    st.header("🎯 Scenario Planning & What-If Analysis")

    # Require dataset
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first in the Data Ingestion tab.")
        return

    st.subheader("🔮 Business Scenario Simulator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Current Baseline**")
        # Use simple dataset-driven baseline from first numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        base_revenue = float(data[numeric_cols[0]].sum()) if numeric_cols else 0.0
        base_costs = float(data[numeric_cols[1]].sum()) if len(numeric_cols) > 1 else base_revenue * 0.7
        base_profit = base_revenue - base_costs
        st.metric("Current Revenue", f"${base_revenue/1_000_000:.1f}M")
        st.metric("Current Costs", f"${base_costs/1_000_000:.1f}M")
        st.metric("Current Profit", f"${base_profit/1_000_000:.1f}M")

    with col2:
        st.markdown("**⚙️ Scenario Parameters**")
        revenue_change = st.slider("Revenue Change (%)", -50, 100, 0)
        cost_change = st.slider("Cost Change (%)", -30, 50, 0)
        market_expansion = st.selectbox("Market Expansion", ["None", "Regional", "National", "International"])

    # Calculate scenario impact
    new_revenue = base_revenue * (1 + revenue_change/100)
    new_costs = base_costs * (1 + cost_change/100)
    new_profit = new_revenue - new_costs

    # Market expansion multiplier
    expansion_multipliers = {"None": 1, "Regional": 1.2, "National": 1.5, "International": 2.0}
    expansion_multiplier = expansion_multipliers[market_expansion]
    new_revenue *= expansion_multiplier
    new_profit = new_revenue - new_costs

    st.subheader("📈 Scenario Results")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Projected Revenue", f"${new_revenue/1_000_000:.1f}M",
                 f"{((new_revenue - base_revenue)/base_revenue)*100:+.1f}%" if base_revenue else "n/a")

    with col4:
        st.metric("Projected Costs", f"${new_costs/1_000_000:.1f}M",
                 f"{((new_costs - base_costs)/base_costs)*100:+.1f}%" if base_costs else "n/a")

    with col5:
        st.metric("Projected Profit", f"${new_profit/1_000_000:.1f}M",
                 f"{((new_profit - base_profit)/base_profit)*100:+.1f}%" if base_profit else "n/a")

    # Scenario comparison chart
    scenarios = ['Current', 'Scenario']
    revenues = [base_revenue/1_000_000, new_revenue/1_000_000]
    costs = [base_costs/1_000_000, new_costs/1_000_000]
    profits = [base_profit/1_000_000, new_profit/1_000_000]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Revenue', x=scenarios, y=revenues))
    fig.add_trace(go.Bar(name='Costs', x=scenarios, y=costs))
    fig.add_trace(go.Bar(name='Profit', x=scenarios, y=profits))

    fig.update_layout(title="Scenario Comparison (Millions $)", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def predictive_insights_tab(ai_core):
    """Predictive insights and forecasting"""
    st.header("🔮 Predictive Insights & Forecasting")
    
    # Forecast configuration
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_metric = st.selectbox("Forecast Metric", 
                                     ["Revenue", "Customer Growth", "Market Share", "Costs"])
        forecast_horizon = st.slider("Forecast Horizon (months)", 1, 24, 6)
    
    with col2:
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        include_seasonality = st.checkbox("Include Seasonality", value=True)
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating predictive insights..."):
            # Mock forecast generation
            import time
            time.sleep(2)
            
            st.success("✅ Forecast generated successfully!")
            
            # Generate mock forecast data
            dates = pd.date_range('2024-01-01', periods=forecast_horizon, freq='M')
            base_value = 2.4 if forecast_metric == "Revenue" else 1000
            
            # Add trend and seasonality
            trend = np.linspace(0, 0.2, forecast_horizon)
            seasonality = 0.1 * np.sin(2 * np.pi * np.arange(forecast_horizon) / 12) if include_seasonality else 0
            noise = np.random.normal(0, 0.05, forecast_horizon)
            
            forecast_values = base_value * (1 + trend + seasonality + noise)
            lower_bound = forecast_values * 0.9
            upper_bound = forecast_values * 1.1
            
            # Create forecast visualization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{confidence_level}% Confidence Interval',
                fillcolor='rgba(0,100,80,0.2)'
            ))
            
            fig.update_layout(
                title=f"{forecast_metric} Forecast - Next {forecast_horizon} Months",
                xaxis_title="Date",
                yaxis_title=forecast_metric,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.subheader("🧠 Key Predictive Insights")
            
            growth_rate = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
            
            insights = [
                f"📈 Expected {growth_rate:+.1f}% growth over {forecast_horizon} months",
                f"🎯 Peak performance expected in month {np.argmax(forecast_values) + 1}",
                f"📊 Average monthly growth rate: {growth_rate/forecast_horizon:.1f}%",
                f"🔍 Confidence level: {confidence_level}% - High reliability forecast",
                f"⚠️ Key risk factors: Market volatility, seasonal variations"
            ]
            
            for insight in insights:
                st.info(insight)

def privacy_security_tab():
    """Privacy and security dashboard"""
    st.header("🛡️ Privacy & Security Dashboard")
    
    # Security metrics
    st.subheader("🔒 Security Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Security Score", "98%", "↗️ 2%")
        st.metric("Data Encryption", "AES-256", "✅ Active")
    
    with col2:
        st.metric("Access Violations", "0", "↘️ 3")
        st.metric("Compliance Score", "100%", "✅ Compliant")
    
    with col3:
        st.metric("Federated Nodes", "12", "↗️ 2")
        st.metric("Data Anonymization", "Active", "✅ Enabled")
    
    # Federated Learning Status
    st.subheader("🤝 Federated Learning Network")
    
    # Mock federated learning visualization
    nodes_data = pd.DataFrame({
        'Node': ['Node A', 'Node B', 'Node C', 'Node D', 'Node E'],
        'Status': ['Active', 'Active', 'Syncing', 'Active', 'Active'],
        'Data_Points': [15000, 12000, 8000, 20000, 11000],
        'Last_Update': ['2 min ago', '1 min ago', '5 min ago', '1 min ago', '3 min ago']
    })
    
    st.dataframe(nodes_data, use_container_width=True)
    
    # Privacy controls
    st.subheader("🔐 Privacy Controls")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("**Data Anonymization**")
        anonymize_pii = st.checkbox("Anonymize PII", value=True)
        differential_privacy = st.checkbox("Differential Privacy", value=True)
        data_masking = st.checkbox("Data Masking", value=False)
    
    with col5:
        st.markdown("**Access Controls**")
        role_based_access = st.checkbox("Role-based Access", value=True)
        multi_factor_auth = st.checkbox("Multi-factor Authentication", value=True)
        audit_logging = st.checkbox("Audit Logging", value=True)
    
    if st.button("🔄 Update Privacy Settings", type="primary"):
        st.success("✅ Privacy settings updated successfully!")
    
    # Compliance dashboard
    st.subheader("📋 Compliance Dashboard")
    
    compliance_items = [
        {"Regulation": "GDPR", "Status": "✅ Compliant", "Last_Audit": "2024-01-10"},
        {"Regulation": "CCPA", "Status": "✅ Compliant", "Last_Audit": "2024-01-08"},
        {"Regulation": "HIPAA", "Status": "✅ Compliant", "Last_Audit": "2024-01-12"},
        {"Regulation": "SOX", "Status": "⚠️ Review Needed", "Last_Audit": "2023-12-15"}
    ]
    
    compliance_df = pd.DataFrame(compliance_items)
    st.dataframe(compliance_df, use_container_width=True)

def trend_analysis_tab(ai_core):
    """Trend analysis and market intelligence"""
    st.header("📈 Trend Analysis & Market Intelligence")
    
    # Trend categories
    trend_category = st.selectbox("Select Trend Category", 
                                 ["Market Trends", "Customer Behavior", "Product Performance", "Competitive Analysis"])
    
    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox("Time Range", ["Last 30 days", "Last 90 days", "Last 6 months", "Last year"])
    with col2:
        granularity = st.selectbox("Granularity", ["Daily", "Weekly", "Monthly"])
    
    # Generate trend analysis
    if st.button("📊 Analyze Trends", type="primary"):
        with st.spinner("Analyzing trends..."):
            # Mock trend analysis
            st.success("✅ Trend analysis completed!")
            
            # Generate mock trend data
            if time_range == "Last 30 days":
                periods = 30
                freq = 'D'
            elif time_range == "Last 90 days":
                periods = 90
                freq = 'D'
            elif time_range == "Last 6 months":
                periods = 6
                freq = 'M'
            else:
                periods = 12
                freq = 'M'
            
            dates = pd.date_range(end='2024-01-15', periods=periods, freq=freq)
            
            # Generate different trend patterns based on category
            if trend_category == "Market Trends":
                values = 100 + np.cumsum(np.random.randn(periods) * 2)
                title = "Market Growth Index"
            elif trend_category == "Customer Behavior":
                values = 50 + 20 * np.sin(2 * np.pi * np.arange(periods) / 12) + np.cumsum(np.random.randn(periods))
                title = "Customer Engagement Score"
            elif trend_category == "Product Performance":
                values = 80 + np.cumsum(np.random.randn(periods) * 1.5)
                title = "Product Performance Index"
            else:
                values = 60 + np.cumsum(np.random.randn(periods) * 1.8)
                title = "Competitive Position Score"
            
            # Create trend visualization
            fig = px.line(x=dates, y=values, title=f"{title} - {time_range}")
            fig.update_traces(line_color='#1f77b4', line_width=3)
            fig.add_hline(y=np.mean(values), line_dash="dash", line_color="red", 
                         annotation_text="Average")
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend insights
            st.subheader("🧠 Trend Insights")
            
            trend_direction = "upward" if values[-1] > values[0] else "downward"
            volatility = "high" if np.std(values) > np.mean(values) * 0.2 else "low"
            
            insights = [
                f"📈 Overall trend is {trend_direction} with {((values[-1] - values[0])/values[0]*100):+.1f}% change",
                f"📊 Volatility is {volatility} with standard deviation of {np.std(values):.1f}",
                f"🎯 Peak value: {np.max(values):.1f} on {dates[np.argmax(values)].strftime('%Y-%m-%d')}",
                f"📉 Lowest value: {np.min(values):.1f} on {dates[np.argmin(values)].strftime('%Y-%m-%d')}",
                f"📊 Current momentum: {'Positive' if values[-1] > values[-5] else 'Negative'}"
            ]
            
            for insight in insights:
                st.info(insight)

def realtime_alerts_tab():
    """Real-time alerts and notifications (requires uploaded data)"""
    st.header("⚡ Real-time Alerts & Notifications")

    # Require dataset
    data = get_active_data()
    if data is None:
        st.warning("⚠️ Please upload or generate data first in the Data Ingestion tab.")
        return

    # Alert configuration
    st.subheader("🔔 Alert Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Performance Alerts**")
        revenue_threshold = st.number_input("Revenue Drop Alert (%)", value=10)
        customer_churn_threshold = st.number_input("Churn Rate Alert (%)", value=5)

    with col2:
        st.markdown("**System Alerts**")
        data_quality_threshold = st.number_input("Data Quality Alert (%)", value=90)
        processing_delay_threshold = st.number_input("Processing Delay Alert (min)", value=30)

    # Notification preferences
    st.subheader("📱 Notification Preferences")

    col3, col4 = st.columns(2)

    with col3:
        email_alerts = st.checkbox("Email Notifications", value=True)
        sms_alerts = st.checkbox("SMS Notifications", value=False)

    with col4:
        dashboard_alerts = st.checkbox("Dashboard Notifications", value=True)
        slack_integration = st.checkbox("Slack Integration", value=False)

    if st.button("💾 Save Alert Settings", type="primary"):
        st.success("✅ Alert settings saved successfully!")

    # Current alerts derived from dataset
    st.subheader("🚨 Current Alerts")

    # Compute simple health metrics from the dataset
    total_cells = len(data) * len(data.columns)
    missing_cells = int(data.isnull().sum().sum())
    data_quality = 0 if total_cells == 0 else (1 - missing_cells / max(total_cells, 1)) * 100

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    current_alerts = []

    # Data quality alert
    if data_quality < data_quality_threshold:
        dq_type = "Critical" if data_quality < max(50, data_quality_threshold - 20) else "Warning"
        current_alerts.append({
            "Time": current_time,
            "Type": dq_type,
            "Message": f"Data quality {data_quality:.1f}% below threshold {data_quality_threshold}%",
            "Status": "Active",
        })

    # Revenue drop alert using first numeric column as proxy
    if numeric_cols:
        kpi_col = numeric_cols[0]
        mean_val = float(data[kpi_col].mean())
        last_val = float(data[kpi_col].iloc[-1]) if len(data) > 0 else mean_val
        drop_pct = 0.0 if mean_val == 0 else ((mean_val - last_val) / abs(mean_val)) * 100
        if drop_pct >= revenue_threshold:
            current_alerts.append({
                "Time": current_time,
                "Type": "Critical" if drop_pct >= revenue_threshold * 1.5 else "Warning",
                "Message": f"Revenue proxy '{kpi_col}' dropped {drop_pct:.1f}% vs mean",
                "Status": "Active",
            })
    
    # Informational alert when data is present
    current_alerts.append({
        "Time": current_time,
        "Type": "Info",
        "Message": "New data batch available and monitored",
        "Status": "Resolved",
    })

    # Render alerts
    for alert in current_alerts:
        if alert["Type"] == "Critical":
            st.error(f"🔴 {alert['Time']} - {alert['Message']} ({alert['Status']})")
        elif alert["Type"] == "Warning":
            st.warning(f"🟡 {alert['Time']} - {alert['Message']} ({alert['Status']})")
        elif alert["Type"] == "Success":
            st.success(f"🟢 {alert['Time']} - {alert['Message']} ({alert['Status']})")
        else:
            st.info(f"🔵 {alert['Time']} - {alert['Message']} ({alert['Status']})")

    # Alert analytics
    st.subheader("📊 Alert Analytics")

    col5, col6, col7 = st.columns(3)

    total_alerts = len(current_alerts)
    critical_alerts = sum(1 for a in current_alerts if a["Type"] == "Critical")

    with col5:
        st.metric("Total Alerts", str(total_alerts))
    with col6:
        st.metric("Critical Alerts", str(critical_alerts))
    with col7:
        st.metric("Resolution Time", "15 min")

def main():
    """Main application function with role-based routing"""
    # Initialize session state
    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = []
    
    # Check if user has selected a role
    if 'user_role' not in st.session_state:
        role_selection_page()
        return
    
    # Route to appropriate interface based on role
    if st.session_state.user_role == "data_scientist":
        data_scientist_interface()
    elif st.session_state.user_role == "business_user":
        business_user_interface()
    else:
        # Fallback to role selection
        del st.session_state.user_role
        st.rerun()

if __name__ == "__main__":
    main()