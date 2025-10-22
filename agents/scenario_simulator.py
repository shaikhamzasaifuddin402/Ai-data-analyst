"""
Scenario Simulation Sandbox for What-If Analysis and Predictive Modeling
Advanced simulation engine with interactive parameter adjustment and real-time predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ScenarioSimulator:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.simulation_history = []
        
    def prepare_data(self, df, target_column):
        """Prepare data for modeling with proper encoding and scaling"""
        try:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            
            X_processed = X.copy()
            
            # Encode categorical variables
            for col in categorical_cols:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X_processed[col] = self.encoders[col].fit_transform(X[col].astype(str))
                else:
                    X_processed[col] = self.encoders[col].transform(X[col].astype(str))
            
            # Scale numerical variables
            if len(numerical_cols) > 0:
                if 'numerical' not in self.scalers:
                    self.scalers['numerical'] = StandardScaler()
                    X_processed[numerical_cols] = self.scalers['numerical'].fit_transform(X[numerical_cols])
                else:
                    X_processed[numerical_cols] = self.scalers['numerical'].transform(X[numerical_cols])
            
            return X_processed, y
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None
    
    def train_models(self, X, y):
        """Train multiple models for ensemble predictions"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Define models
            model_configs = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0)
            }
            
            model_performance = {}
            
            # Train each model
            for name, model in model_configs.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                self.models[name] = model
                model_performance[name] = {
                    'MSE': mse,
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': np.sqrt(mse)
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            return model_performance
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return {}
    
    def run_scenario(self, base_data, modifications, model_name='Random Forest'):
        """Run what-if scenario with modified parameters"""
        try:
            if model_name not in self.models:
                return None
            
            # Create scenario data
            scenario_data = base_data.copy()
            
            # Apply modifications
            for feature, value in modifications.items():
                if feature in scenario_data.columns:
                    scenario_data[feature] = value
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(scenario_data)
            
            # Store in history
            scenario_result = {
                'modifications': modifications,
                'prediction': prediction[0] if len(prediction) == 1 else prediction.mean(),
                'model': model_name,
                'timestamp': pd.Timestamp.now()
            }
            self.simulation_history.append(scenario_result)
            
            return scenario_result
            
        except Exception as e:
            st.error(f"Error running scenario: {str(e)}")
            return None
    
    def monte_carlo_simulation(self, base_data, feature_ranges, n_simulations=1000, model_name='Random Forest'):
        """Run Monte Carlo simulation with random parameter variations"""
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            results = []
            
            for _ in range(n_simulations):
                scenario_data = base_data.copy()
                
                # Generate random values within specified ranges
                for feature, (min_val, max_val) in feature_ranges.items():
                    if feature in scenario_data.columns:
                        random_value = np.random.uniform(min_val, max_val)
                        scenario_data[feature] = random_value
                
                # Make prediction
                prediction = model.predict(scenario_data)
                results.append(prediction[0] if len(prediction) == 1 else prediction.mean())
            
            return np.array(results)
            
        except Exception as e:
            st.error(f"Error in Monte Carlo simulation: {str(e)}")
            return None
    
    def sensitivity_analysis(self, base_data, target_feature, variation_range=0.2, steps=20, model_name='Random Forest'):
        """Perform sensitivity analysis for a specific feature"""
        try:
            if model_name not in self.models or target_feature not in base_data.columns:
                return None
            
            model = self.models[model_name]
            base_value = base_data[target_feature].iloc[0]
            
            # Create variation range
            min_val = base_value * (1 - variation_range)
            max_val = base_value * (1 + variation_range)
            values = np.linspace(min_val, max_val, steps)
            
            predictions = []
            for value in values:
                scenario_data = base_data.copy()
                scenario_data[target_feature] = value
                prediction = model.predict(scenario_data)
                predictions.append(prediction[0] if len(prediction) == 1 else prediction.mean())
            
            return values, np.array(predictions)
            
        except Exception as e:
            st.error(f"Error in sensitivity analysis: {str(e)}")
            return None, None

def render_scenario_simulator():
    """Render the Scenario Simulation Sandbox interface"""
    st.header("🎯 Scenario Simulation Sandbox")
    st.markdown("**Advanced What-If Analysis & Predictive Modeling**")
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = ScenarioSimulator()
    
    simulator = st.session_state.simulator
    
    # Data upload section
    st.subheader("📊 Data Setup")
    uploaded_file = st.file_uploader("Upload dataset for simulation", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
            
            # Target variable selection
            target_column = st.selectbox("Select target variable for prediction:", df.columns)
            
            if target_column:
                # Prepare data and train models
                if st.button("🚀 Initialize Models"):
                    with st.spinner("Training prediction models..."):
                        X, y = simulator.prepare_data(df, target_column)
                        if X is not None and y is not None:
                            performance = simulator.train_models(X, y)
                            
                            if performance:
                                st.success("Models trained successfully!")
                                
                                # Display model performance
                                st.subheader("🎯 Model Performance")
                                perf_df = pd.DataFrame(performance).T
                                st.dataframe(perf_df.round(4))
                                
                                # Store in session state
                                st.session_state.df = df
                                st.session_state.X = X
                                st.session_state.y = y
                                st.session_state.target_column = target_column
                
                # Simulation interface (only show if models are trained)
                if hasattr(st.session_state, 'X') and len(simulator.models) > 0:
                    st.markdown("---")
                    
                    # Simulation tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "🎮 Interactive Simulation", 
                        "🎲 Monte Carlo Analysis", 
                        "📈 Sensitivity Analysis", 
                        "📊 Simulation History"
                    ])
                    
                    with tab1:
                        render_interactive_simulation(simulator, st.session_state.X, st.session_state.df)
                    
                    with tab2:
                        render_monte_carlo_analysis(simulator, st.session_state.X)
                    
                    with tab3:
                        render_sensitivity_analysis(simulator, st.session_state.X)
                    
                    with tab4:
                        render_simulation_history(simulator)
        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

def render_interactive_simulation(simulator, X, df):
    """Render interactive simulation interface"""
    st.subheader("🎮 Interactive What-If Simulation")
    
    # Model selection
    model_name = st.selectbox("Select prediction model:", list(simulator.models.keys()))
    
    # Feature modification interface
    st.markdown("**Adjust Parameters:**")
    modifications = {}
    
    # Create columns for better layout
    cols = st.columns(3)
    
    for i, feature in enumerate(X.columns):
        col_idx = i % 3
        with cols[col_idx]:
            if X[feature].dtype in ['int64', 'float64']:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                current_val = float(X[feature].mean())
                
                new_val = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=current_val,
                    key=f"slider_{feature}"
                )
                modifications[feature] = new_val
            else:
                unique_vals = X[feature].unique()
                current_val = unique_vals[0]
                new_val = st.selectbox(f"{feature}", unique_vals, key=f"select_{feature}")
                modifications[feature] = new_val
    
    # Run simulation
    if st.button("🚀 Run Simulation"):
        # Create base data (using mean values)
        base_data = pd.DataFrame([X.mean()], columns=X.columns)
        
        # Run scenario
        result = simulator.run_scenario(base_data, modifications, model_name)
        
        if result:
            st.success(f"**Prediction Result: {result['prediction']:.4f}**")
            
            # Show modifications
            st.markdown("**Applied Modifications:**")
            for feature, value in modifications.items():
                st.write(f"• {feature}: {value}")

def render_monte_carlo_analysis(simulator, X):
    """Render Monte Carlo analysis interface"""
    st.subheader("🎲 Monte Carlo Simulation")
    
    # Model selection
    model_name = st.selectbox("Select model for Monte Carlo:", list(simulator.models.keys()), key="mc_model")
    
    # Number of simulations
    n_simulations = st.slider("Number of simulations:", 100, 5000, 1000)
    
    # Feature range selection
    st.markdown("**Define Parameter Ranges:**")
    feature_ranges = {}
    
    selected_features = st.multiselect("Select features to vary:", X.columns)
    
    for feature in selected_features:
        if X[feature].dtype in ['int64', 'float64']:
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input(f"{feature} - Min:", value=float(X[feature].min()))
            with col2:
                max_val = st.number_input(f"{feature} - Max:", value=float(X[feature].max()))
            feature_ranges[feature] = (min_val, max_val)
    
    # Run Monte Carlo simulation
    if st.button("🎲 Run Monte Carlo Simulation") and feature_ranges:
        with st.spinner("Running Monte Carlo simulation..."):
            base_data = pd.DataFrame([X.mean()], columns=X.columns)
            results = simulator.monte_carlo_simulation(base_data, feature_ranges, n_simulations, model_name)
            
            if results is not None:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mean Prediction", f"{results.mean():.4f}")
                    st.metric("Std Deviation", f"{results.std():.4f}")
                
                with col2:
                    st.metric("Min Prediction", f"{results.min():.4f}")
                    st.metric("Max Prediction", f"{results.max():.4f}")
                
                # Histogram
                fig = px.histogram(x=results, nbins=50, title="Monte Carlo Simulation Results")
                fig.update_layout(xaxis_title="Predicted Value", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
                
                # Percentiles
                percentiles = [5, 25, 50, 75, 95]
                perc_values = np.percentile(results, percentiles)
                perc_df = pd.DataFrame({
                    'Percentile': [f"{p}%" for p in percentiles],
                    'Value': perc_values
                })
                st.dataframe(perc_df)

def render_sensitivity_analysis(simulator, X):
    """Render sensitivity analysis interface"""
    st.subheader("📈 Sensitivity Analysis")
    
    # Model and feature selection
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("Select model:", list(simulator.models.keys()), key="sens_model")
    with col2:
        target_feature = st.selectbox("Select feature to analyze:", X.columns)
    
    # Analysis parameters
    variation_range = st.slider("Variation range (±%):", 0.1, 0.5, 0.2)
    steps = st.slider("Number of steps:", 10, 50, 20)
    
    # Run sensitivity analysis
    if st.button("📈 Run Sensitivity Analysis"):
        with st.spinner("Running sensitivity analysis..."):
            base_data = pd.DataFrame([X.mean()], columns=X.columns)
            values, predictions = simulator.sensitivity_analysis(
                base_data, target_feature, variation_range, steps, model_name
            )
            
            if values is not None and predictions is not None:
                # Create sensitivity plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=values,
                    y=predictions,
                    mode='lines+markers',
                    name='Sensitivity Curve',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title=f"Sensitivity Analysis: {target_feature}",
                    xaxis_title=target_feature,
                    yaxis_title="Predicted Value",
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate sensitivity metrics
                sensitivity = (predictions.max() - predictions.min()) / (values.max() - values.min())
                st.metric("Sensitivity Coefficient", f"{sensitivity:.4f}")

def render_simulation_history(simulator):
    """Render simulation history interface"""
    st.subheader("📊 Simulation History")
    
    if simulator.simulation_history:
        # Convert history to DataFrame
        history_data = []
        for i, sim in enumerate(simulator.simulation_history):
            row = {
                'Simulation #': i + 1,
                'Model': sim['model'],
                'Prediction': sim['prediction'],
                'Timestamp': sim['timestamp']
            }
            # Add modifications
            for feature, value in sim['modifications'].items():
                row[f"Modified_{feature}"] = value
            history_data.append(row)
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df)
        
        # Plot prediction trends
        if len(history_data) > 1:
            fig = px.line(
                history_df, 
                x='Simulation #', 
                y='Prediction',
                color='Model',
                title="Prediction Trends Across Simulations"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            simulator.simulation_history = []
            st.rerun()
    else:
        st.info("No simulation history available. Run some simulations to see results here.")