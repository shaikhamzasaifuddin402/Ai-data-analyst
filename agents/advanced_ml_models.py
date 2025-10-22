"""
Advanced ML Models Module
Comprehensive machine learning models including ensemble methods, deep learning, and AutoML
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           mean_squared_error, mean_absolute_error, r2_score, 
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)

# Ensemble Methods
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            AdaBoostClassifier, AdaBoostRegressor,
                            ExtraTreesClassifier, ExtraTreesRegressor,
                            VotingClassifier, VotingRegressor,
                            BaggingClassifier, BaggingRegressor)

# Advanced Models
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, 
                                ElasticNet, SGDClassifier, SGDRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Dimensionality Reduction
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE

# Model Selection and Validation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Try to import additional libraries (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ModelRegistry:
    """Registry of available ML models"""
    
    @staticmethod
    def get_classification_models():
        """Get available classification models"""
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'SGD Classifier': SGDClassifier(random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        return models
    
    @staticmethod
    def get_regression_models():
        """Get available regression models"""
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42),
            'SVM': SVR(),
            'Neural Network': MLPRegressor(random_state=42, max_iter=1000),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Elastic Net': ElasticNet(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'SGD Regressor': SGDRegressor(random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        return models
    
    @staticmethod
    def get_clustering_models():
        """Get available clustering models"""
        return {
            'K-Means': KMeans(random_state=42),
            'DBSCAN': DBSCAN(),
            'Agglomerative': AgglomerativeClustering(),
            'Gaussian Mixture': GaussianMixture(random_state=42)
        }

class HyperparameterOptimizer:
    """Hyperparameter optimization for ML models"""
    
    def __init__(self):
        self.param_grids = self._get_parameter_grids()
    
    def _get_parameter_grids(self):
        """Get parameter grids for different models"""
        return {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'RandomForestRegressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'SVR': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'MLPRegressor': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    
    def optimize_hyperparameters(self, model, X_train, y_train, 
                                search_type='grid', cv_folds=5, n_iter=50):
        """Optimize hyperparameters for a given model"""
        model_name = type(model).__name__
        
        if model_name not in self.param_grids:
            return model, {}
        
        param_grid = self.param_grids[model_name]
        
        try:
            if search_type == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=cv_folds, 
                    scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
                    n_jobs=-1, verbose=0
                )
            else:  # randomized search
                search = RandomizedSearchCV(
                    model, param_grid, cv=cv_folds, n_iter=n_iter,
                    scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
                    n_jobs=-1, verbose=0, random_state=42
                )
            
            search.fit(X_train, y_train)
            
            return search.best_estimator_, {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
        
        except Exception as e:
            st.warning(f"Hyperparameter optimization failed for {model_name}: {str(e)}")
            return model, {}

class EnsembleBuilder:
    """Build ensemble models"""
    
    def __init__(self):
        self.ensemble_methods = ['voting', 'bagging', 'stacking']
    
    def create_voting_ensemble(self, models_dict, task_type='classification'):
        """Create voting ensemble"""
        estimators = [(name, model) for name, model in models_dict.items()]
        
        if task_type == 'classification':
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        return ensemble
    
    def create_bagging_ensemble(self, base_model, n_estimators=10):
        """Create bagging ensemble"""
        if hasattr(base_model, 'predict_proba'):
            ensemble = BaggingClassifier(
                base_estimator=base_model, 
                n_estimators=n_estimators, 
                random_state=42
            )
        else:
            ensemble = BaggingRegressor(
                base_estimator=base_model, 
                n_estimators=n_estimators, 
                random_state=42
            )
        
        return ensemble

class AutoMLEngine:
    """Automated Machine Learning Engine"""
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.optimizer = HyperparameterOptimizer()
        self.ensemble_builder = EnsembleBuilder()
        self.results = {}
    
    def auto_train_models(self, X, y, task_type='auto', test_size=0.2, 
                         optimize_hyperparams=False, create_ensemble=False):
        """Automatically train multiple models and compare performance"""
        
        # Detect task type if auto
        if task_type == 'auto':
            if y.dtype == 'object' or y.nunique() <= 20:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Get appropriate models
        if task_type == 'classification':
            models = self.model_registry.get_classification_models()
        else:
            models = self.model_registry.get_regression_models()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task_type == 'classification' and y.nunique() > 1 else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        trained_models = {}
        
        # Train each model
        for name, model in models.items():
            try:
                start_time = time.time()
                
                # Optimize hyperparameters if requested
                if optimize_hyperparams:
                    model, opt_results = self.optimizer.optimize_hyperparameters(
                        model, X_train_scaled, y_train
                    )
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                if task_type == 'classification':
                    metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test_scaled)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'training_time': training_time,
                    'hyperparams_optimized': optimize_hyperparams
                }
                
                trained_models[name] = model
                
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        # Create ensemble if requested
        if create_ensemble and len(trained_models) >= 2:
            try:
                ensemble = self.ensemble_builder.create_voting_ensemble(
                    trained_models, task_type
                )
                
                start_time = time.time()
                ensemble.fit(X_train_scaled, y_train)
                y_pred_ensemble = ensemble.predict(X_test_scaled)
                
                if task_type == 'classification':
                    ensemble_metrics = self._calculate_classification_metrics(
                        y_test, y_pred_ensemble, ensemble, X_test_scaled
                    )
                else:
                    ensemble_metrics = self._calculate_regression_metrics(y_test, y_pred_ensemble)
                
                training_time = time.time() - start_time
                
                results['Voting Ensemble'] = {
                    'model': ensemble,
                    'metrics': ensemble_metrics,
                    'training_time': training_time,
                    'hyperparams_optimized': False
                }
            
            except Exception as e:
                st.warning(f"Failed to create ensemble: {str(e)}")
        
        self.results = results
        return results, scaler, (X_train, X_test, y_train, y_test)
    
    def _calculate_classification_metrics(self, y_true, y_pred, model, X_test):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model based on specified metric"""
        if not self.results:
            return None, None
        
        best_score = -np.inf if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score'] else np.inf
        best_model_name = None
        
        for name, result in self.results.items():
            if metric in result['metrics']:
                score = result['metrics'][metric]
                
                if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']:
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                else:  # Lower is better for MSE, MAE, RMSE
                    if score < best_score:
                        best_score = score
                        best_model_name = name
        
        if best_model_name:
            return best_model_name, self.results[best_model_name]
        
        return None, None

class DeepLearningSimulator:
    """Simulate deep learning capabilities"""
    
    def __init__(self):
        self.architectures = {
            'Dense Neural Network': 'Multi-layer perceptron with dense connections',
            'Convolutional Neural Network': 'CNN for image-like data patterns',
            'Recurrent Neural Network': 'RNN for sequential data analysis',
            'Long Short-Term Memory': 'LSTM for long sequence dependencies',
            'Transformer': 'Attention-based architecture for complex patterns',
            'Autoencoder': 'Unsupervised learning for dimensionality reduction',
            'Generative Adversarial Network': 'GAN for synthetic data generation'
        }
    
    def simulate_deep_learning(self, X, y, architecture='Dense Neural Network'):
        """Simulate deep learning training"""
        
        # Use MLPClassifier/MLPRegressor as a proxy for deep learning
        if y.dtype == 'object' or y.nunique() <= 20:
            # Classification
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            task_type = 'classification'
        else:
            # Regression
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            task_type = 'regression'
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return {
            'model': model,
            'metrics': metrics,
            'architecture': architecture,
            'task_type': task_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'scaler': scaler
        }

def render_advanced_ml_models():
    """Render the Advanced ML Models interface"""
    st.header("🧠 Advanced ML Models & AutoML")
    st.markdown("**Comprehensive Machine Learning with Ensemble Methods, Deep Learning & AutoML**")
    
    # Check if data is available
    if st.session_state.data is None:
        st.info("📤 Please upload data in the **Data Upload** tab first.")
        return
    
    # Use processed data if available
    data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    
    # Main functionality tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AutoML Engine",
        "🏗️ Ensemble Methods",
        "🧠 Deep Learning",
        "📊 Model Comparison",
        "🔧 Advanced Configuration"
    ])
    
    with tab1:
        render_automl_engine(data)
    
    with tab2:
        render_ensemble_methods(data)
    
    with tab3:
        render_deep_learning(data)
    
    with tab4:
        render_model_comparison(data)
    
    with tab5:
        render_advanced_configuration(data)

def render_automl_engine(data):
    """Render AutoML engine interface"""
    st.subheader("🤖 Automated Machine Learning Engine")
    st.markdown("Automatically train and compare multiple ML models")
    
    # Initialize AutoML engine
    if 'automl_engine' not in st.session_state:
        st.session_state.automl_engine = AutoMLEngine()
    
    automl = st.session_state.automl_engine
    
    # Configuration
    st.markdown("**AutoML Configuration:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_column = st.selectbox("Target Column:", data.columns)
    
    with col2:
        task_type = st.selectbox("Task Type:", ["auto", "classification", "regression"])
    
    with col3:
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
    
    # Advanced options
    st.markdown("**Advanced Options:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimize_hyperparams = st.checkbox("Optimize Hyperparameters", value=False)
    
    with col2:
        create_ensemble = st.checkbox("Create Ensemble Models", value=True)
    
    # Start AutoML
    if st.button("🚀 Start AutoML Training"):
        if target_column in data.columns:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle target variable for classification
            if task_type == 'classification' or (task_type == 'auto' and y.dtype == 'object'):
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
            
            with st.spinner("Training multiple ML models... This may take a few minutes."):
                try:
                    results, scaler, data_splits = automl.auto_train_models(
                        X, y, task_type, test_size, optimize_hyperparams, create_ensemble
                    )
                    
                    if results:
                        st.success(f"✅ AutoML completed! Trained {len(results)} models.")
                        
                        # Display results
                        st.markdown("**Model Performance Comparison:**")
                        
                        # Create results dataframe
                        results_data = []
                        for name, result in results.items():
                            row = {'Model': name}
                            row.update(result['metrics'])
                            row['Training Time (s)'] = f"{result['training_time']:.2f}"
                            row['Hyperparams Optimized'] = '✅' if result['hyperparams_optimized'] else '❌'
                            results_data.append(row)
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Best model
                        primary_metric = 'accuracy' if task_type == 'classification' else 'r2_score'
                        best_name, best_result = automl.get_best_model(primary_metric)
                        
                        if best_name:
                            st.success(f"🏆 **Best Model:** {best_name}")
                            st.info(f"**Best {primary_metric.upper()}:** {best_result['metrics'][primary_metric]:.4f}")
                        
                        # Visualize results
                        if len(results) > 1:
                            # Performance comparison chart
                            fig = px.bar(
                                results_df,
                                x='Model',
                                y=primary_metric if primary_metric in results_df.columns else list(results_df.columns)[1],
                                title=f'Model Performance Comparison ({primary_metric.upper()})',
                                text=primary_metric if primary_metric in results_df.columns else list(results_df.columns)[1]
                            )
                            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Store results in session state
                        st.session_state.automl_results = results
                        st.session_state.automl_scaler = scaler
                        st.session_state.automl_data_splits = data_splits
                    
                    else:
                        st.error("No models were successfully trained.")
                
                except Exception as e:
                    st.error(f"AutoML training failed: {str(e)}")
        
        else:
            st.error("Please select a valid target column.")
    
    # Show previous results if available
    if hasattr(st.session_state, 'automl_results') and st.session_state.automl_results:
        st.markdown("---")
        st.subheader("📊 Previous AutoML Results")
        
        results = st.session_state.automl_results
        
        # Model selection for detailed analysis
        selected_model = st.selectbox("Select Model for Detailed Analysis:", 
                                    list(results.keys()))
        
        if selected_model and selected_model in results:
            model_result = results[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Metrics:**")
                for metric, value in model_result['metrics'].items():
                    st.metric(metric.upper(), f"{value:.4f}")
            
            with col2:
                st.markdown("**Model Information:**")
                st.info(f"**Training Time:** {model_result['training_time']:.2f} seconds")
                st.info(f"**Hyperparameters Optimized:** {'Yes' if model_result['hyperparams_optimized'] else 'No'}")
                st.info(f"**Model Type:** {type(model_result['model']).__name__}")

def render_ensemble_methods(data):
    """Render ensemble methods interface"""
    st.subheader("🏗️ Ensemble Methods")
    st.markdown("Build powerful ensemble models combining multiple algorithms")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox("Target Column:", data.columns, key="ensemble_target")
    
    with col2:
        ensemble_type = st.selectbox("Ensemble Type:", 
                                   ["Voting Classifier", "Voting Regressor", "Random Forest", 
                                    "Gradient Boosting", "AdaBoost", "Extra Trees"])
    
    # Model selection for voting ensemble
    if "Voting" in ensemble_type:
        st.markdown("**Select Base Models:**")
        
        if "Classifier" in ensemble_type:
            available_models = ModelRegistry.get_classification_models()
        else:
            available_models = ModelRegistry.get_regression_models()
        
        selected_models = st.multiselect("Choose models for ensemble:", 
                                       list(available_models.keys()),
                                       default=list(available_models.keys())[:3])
    
    # Train ensemble
    if st.button("🚀 Train Ensemble Model"):
        if target_column in data.columns:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Detect task type
            task_type = 'classification' if y.dtype == 'object' or y.nunique() <= 20 else 'regression'
            
            # Handle target variable for classification
            if task_type == 'classification' and y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            with st.spinner("Training ensemble model..."):
                try:
                    # Create ensemble model
                    if ensemble_type == "Voting Classifier" and selected_models:
                        models_dict = {name: ModelRegistry.get_classification_models()[name] 
                                     for name in selected_models}
                        ensemble_builder = EnsembleBuilder()
                        model = ensemble_builder.create_voting_ensemble(models_dict, 'classification')
                    
                    elif ensemble_type == "Voting Regressor" and selected_models:
                        models_dict = {name: ModelRegistry.get_regression_models()[name] 
                                     for name in selected_models}
                        ensemble_builder = EnsembleBuilder()
                        model = ensemble_builder.create_voting_ensemble(models_dict, 'regression')
                    
                    elif ensemble_type == "Random Forest":
                        if task_type == 'classification':
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    elif ensemble_type == "Gradient Boosting":
                        if task_type == 'classification':
                            model = GradientBoostingClassifier(random_state=42)
                        else:
                            model = GradientBoostingRegressor(random_state=42)
                    
                    elif ensemble_type == "AdaBoost":
                        if task_type == 'classification':
                            model = AdaBoostClassifier(random_state=42)
                        else:
                            model = AdaBoostRegressor(random_state=42)
                    
                    elif ensemble_type == "Extra Trees":
                        if task_type == 'classification':
                            model = ExtraTreesClassifier(n_estimators=100, random_state=42)
                        else:
                            model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                    
                    # Train model
                    start_time = time.time()
                    model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    if task_type == 'classification':
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        }
                    else:
                        metrics = {
                            'mse': mean_squared_error(y_test, y_pred),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2_score': r2_score(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                    
                    # Display results
                    st.success(f"✅ {ensemble_type} trained successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Performance Metrics:**")
                        for metric, value in metrics.items():
                            st.metric(metric.upper(), f"{value:.4f}")
                    
                    with col2:
                        st.markdown("**Model Information:**")
                        st.info(f"**Training Time:** {training_time:.2f} seconds")
                        st.info(f"**Training Samples:** {len(X_train):,}")
                        st.info(f"**Test Samples:** {len(X_test):,}")
                        st.info(f"**Features:** {X.shape[1]}")
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("**Feature Importance:**")
                        
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Feature Importances'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Ensemble training failed: {str(e)}")

def render_deep_learning(data):
    """Render deep learning interface"""
    st.subheader("🧠 Deep Learning Architectures")
    st.markdown("Simulate advanced deep learning models for complex pattern recognition")
    
    # Initialize deep learning simulator
    dl_simulator = DeepLearningSimulator()
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox("Target Column:", data.columns, key="dl_target")
    
    with col2:
        architecture = st.selectbox("Architecture:", list(dl_simulator.architectures.keys()))
    
    # Architecture description
    st.info(f"**{architecture}:** {dl_simulator.architectures[architecture]}")
    
    # Advanced configuration
    with st.expander("🔧 Advanced Configuration"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hidden_layers = st.text_input("Hidden Layers (comma-separated):", "100,50,25")
        
        with col2:
            activation = st.selectbox("Activation Function:", ["relu", "tanh", "sigmoid"])
        
        with col3:
            learning_rate = st.selectbox("Learning Rate:", ["constant", "adaptive"])
    
    # Train deep learning model
    if st.button("🚀 Train Deep Learning Model"):
        if target_column in data.columns:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle target variable for classification
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            with st.spinner("Training deep learning model... This may take a while."):
                try:
                    result = dl_simulator.simulate_deep_learning(X, y, architecture)
                    
                    st.success(f"✅ {architecture} trained successfully!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Performance Metrics:**")
                        for metric, value in result['metrics'].items():
                            st.metric(metric.upper(), f"{value:.4f}")
                    
                    with col2:
                        st.markdown("**Model Architecture:**")
                        st.info(f"**Architecture:** {result['architecture']}")
                        st.info(f"**Task Type:** {result['task_type'].title()}")
                        st.info(f"**Training Samples:** {result['training_samples']:,}")
                        st.info(f"**Test Samples:** {result['test_samples']:,}")
                        st.info(f"**Input Features:** {result['features']}")
                    
                    # Training visualization (simulated)
                    st.markdown("**Training Progress (Simulated):**")
                    
                    # Create simulated training curves
                    epochs = np.arange(1, 101)
                    
                    if result['task_type'] == 'classification':
                        # Simulated accuracy curve
                        train_acc = 0.5 + 0.4 * (1 - np.exp(-epochs/20)) + np.random.normal(0, 0.02, len(epochs))
                        val_acc = 0.5 + 0.35 * (1 - np.exp(-epochs/25)) + np.random.normal(0, 0.03, len(epochs))
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Training Accuracy', mode='lines'))
                        fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation Accuracy', mode='lines'))
                        fig.update_layout(title='Model Training Progress', xaxis_title='Epoch', yaxis_title='Accuracy')
                        
                    else:
                        # Simulated loss curve
                        train_loss = 1.0 * np.exp(-epochs/15) + np.random.normal(0, 0.05, len(epochs))
                        val_loss = 1.2 * np.exp(-epochs/20) + np.random.normal(0, 0.07, len(epochs))
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss', mode='lines'))
                        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', mode='lines'))
                        fig.update_layout(title='Model Training Progress', xaxis_title='Epoch', yaxis_title='Loss')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Deep learning training failed: {str(e)}")
    
    # Deep learning information
    st.markdown("---")
    st.markdown("**🧠 Deep Learning Architectures Overview:**")
    
    for arch, desc in dl_simulator.architectures.items():
        with st.expander(f"📚 {arch}"):
            st.write(desc)
            
            # Add specific use cases
            use_cases = {
                'Dense Neural Network': "Tabular data, feature learning, general classification/regression",
                'Convolutional Neural Network': "Image recognition, computer vision, spatial pattern detection",
                'Recurrent Neural Network': "Time series analysis, sequential data, natural language processing",
                'Long Short-Term Memory': "Long sequence modeling, speech recognition, machine translation",
                'Transformer': "Natural language understanding, attention mechanisms, complex relationships",
                'Autoencoder': "Dimensionality reduction, anomaly detection, data compression",
                'Generative Adversarial Network': "Data generation, image synthesis, data augmentation"
            }
            
            st.markdown(f"**Use Cases:** {use_cases.get(arch, 'Various machine learning tasks')}")

def render_model_comparison(data):
    """Render model comparison interface"""
    st.subheader("📊 Model Performance Comparison")
    st.markdown("Compare different ML models side by side")
    
    # Check if AutoML results are available
    if hasattr(st.session_state, 'automl_results') and st.session_state.automl_results:
        results = st.session_state.automl_results
        
        # Model selection for comparison
        st.markdown("**Select Models to Compare:**")
        selected_models = st.multiselect("Choose models:", list(results.keys()), 
                                       default=list(results.keys())[:3])
        
        if len(selected_models) >= 2:
            # Create comparison dataframe
            comparison_data = []
            for model_name in selected_models:
                result = results[model_name]
                row = {'Model': model_name}
                row.update(result['metrics'])
                row['Training Time'] = result['training_time']
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.markdown("**Model Comparison Table:**")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization options
            st.markdown("**Comparison Visualizations:**")
            
            # Metric selection for visualization
            numeric_columns = [col for col in comparison_df.columns if col != 'Model' and 
                             pd.api.types.is_numeric_dtype(comparison_df[col])]
            
            if numeric_columns:
                selected_metric = st.selectbox("Select metric for visualization:", numeric_columns)
                
                # Bar chart comparison
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y=selected_metric,
                    title=f'Model Comparison: {selected_metric}',
                    text=selected_metric
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart for multiple metrics
                if len(numeric_columns) >= 3:
                    st.markdown("**Multi-Metric Radar Chart:**")
                    
                    # Normalize metrics for radar chart
                    normalized_df = comparison_df.copy()
                    for col in numeric_columns:
                        if col != 'Training Time':  # Higher is better for most metrics
                            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
                        else:  # Lower is better for training time
                            normalized_df[col] = 1 - (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    for _, row in normalized_df.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[row[col] for col in numeric_columns],
                            theta=numeric_columns,
                            fill='toself',
                            name=row['Model']
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Normalized Model Performance Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please select at least 2 models for comparison.")
    
    else:
        st.info("No model results available. Please run AutoML first to generate models for comparison.")
        
        # Show example comparison
        st.markdown("**Example Model Comparison:**")
        
        example_data = {
            'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network', 'SVM'],
            'Accuracy': [0.85, 0.87, 0.82, 0.84],
            'Precision': [0.83, 0.86, 0.80, 0.82],
            'Recall': [0.85, 0.87, 0.82, 0.84],
            'F1 Score': [0.84, 0.86, 0.81, 0.83],
            'Training Time': [2.3, 5.7, 12.1, 8.4]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        # Example visualization
        fig = px.bar(
            example_df,
            x='Model',
            y='Accuracy',
            title='Example Model Accuracy Comparison',
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

def render_advanced_configuration(data):
    """Render advanced configuration interface"""
    st.subheader("🔧 Advanced ML Configuration")
    st.markdown("Fine-tune model parameters and training settings")
    
    # Model selection
    st.markdown("**Model Selection:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_category = st.selectbox("Model Category:", 
                                    ["Classification", "Regression", "Clustering"])
    
    with col2:
        if model_category == "Classification":
            models = ModelRegistry.get_classification_models()
        elif model_category == "Regression":
            models = ModelRegistry.get_regression_models()
        else:
            models = ModelRegistry.get_clustering_models()
        
        selected_model = st.selectbox("Select Model:", list(models.keys()))
    
    # Hyperparameter configuration
    st.markdown("**Hyperparameter Configuration:**")
    
    if selected_model:
        model = models[selected_model]
        model_name = type(model).__name__
        
        # Get default parameters
        params = model.get_params()
        
        # Create parameter inputs based on model type
        if model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider("Number of Estimators:", 10, 500, params.get('n_estimators', 100))
            
            with col2:
                max_depth = st.slider("Max Depth:", 1, 50, params.get('max_depth', 10) if params.get('max_depth') else 10)
            
            with col3:
                min_samples_split = st.slider("Min Samples Split:", 2, 20, params.get('min_samples_split', 2))
        
        elif model_name in ['GradientBoostingClassifier', 'GradientBoostingRegressor']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider("Number of Estimators:", 10, 500, params.get('n_estimators', 100))
            
            with col2:
                learning_rate = st.slider("Learning Rate:", 0.01, 1.0, params.get('learning_rate', 0.1), 0.01)
            
            with col3:
                max_depth = st.slider("Max Depth:", 1, 20, params.get('max_depth', 3))
        
        elif model_name in ['SVC', 'SVR']:
            col1, col2 = st.columns(2)
            
            with col1:
                C = st.slider("Regularization (C):", 0.1, 100.0, params.get('C', 1.0), 0.1)
            
            with col2:
                kernel = st.selectbox("Kernel:", ['rbf', 'poly', 'sigmoid', 'linear'], 
                                    index=['rbf', 'poly', 'sigmoid', 'linear'].index(params.get('kernel', 'rbf')))
        
        elif model_name in ['MLPClassifier', 'MLPRegressor']:
            col1, col2 = st.columns(2)
            
            with col1:
                hidden_layer_sizes = st.text_input("Hidden Layer Sizes (comma-separated):", "100,50")
            
            with col2:
                activation = st.selectbox("Activation:", ['relu', 'tanh', 'logistic'], 
                                        index=['relu', 'tanh', 'logistic'].index(params.get('activation', 'relu')))
    
    # Cross-validation settings
    st.markdown("**Cross-Validation Settings:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider("CV Folds:", 3, 10, 5)
    
    with col2:
        cv_scoring = st.selectbox("Scoring Metric:", 
                                ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] if model_category == "Classification" 
                                else ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'])
    
    # Feature engineering options
    st.markdown("**Feature Engineering:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scaling_method = st.selectbox("Feature Scaling:", 
                                    ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
    
    with col2:
        feature_selection = st.checkbox("Enable Feature Selection")
    
    with col3:
        if feature_selection:
            n_features = st.slider("Number of Features:", 1, min(20, len(data.columns)-1), 10)
    
    # Advanced options
    with st.expander("🔬 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            random_state = st.number_input("Random State:", 0, 1000, 42)
        
        with col2:
            n_jobs = st.selectbox("Parallel Jobs:", [-1, 1, 2, 4, 8], index=0)
        
        # Early stopping for applicable models
        if model_name in ['GradientBoostingClassifier', 'GradientBoostingRegressor', 'MLPClassifier', 'MLPRegressor']:
            early_stopping = st.checkbox("Enable Early Stopping")
            
            if early_stopping:
                validation_fraction = st.slider("Validation Fraction:", 0.1, 0.3, 0.1, 0.05)
    
    # Model training with custom configuration
    if st.button("🚀 Train with Custom Configuration"):
        if model_category != "Clustering":
            target_column = st.selectbox("Target Column:", data.columns, key="custom_target")
            
            if target_column in data.columns:
                # Prepare data
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Handle categorical variables
                categorical_columns = X.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # Handle target variable for classification
                if model_category == "Classification" and y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                
                with st.spinner("Training model with custom configuration..."):
                    try:
                        # Apply feature scaling
                        if scaling_method != "None":
                            if scaling_method == "StandardScaler":
                                scaler = StandardScaler()
                            elif scaling_method == "MinMaxScaler":
                                scaler = MinMaxScaler()
                            else:
                                scaler = RobustScaler()
                            
                            X_scaled = scaler.fit_transform(X)
                            X = pd.DataFrame(X_scaled, columns=X.columns)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
                        
                        # Train model
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        if model_category == "Classification":
                            metrics = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            }
                        else:
                            metrics = {
                                'mse': mean_squared_error(y_test, y_pred),
                                'mae': mean_absolute_error(y_test, y_pred),
                                'r2_score': r2_score(y_test, y_pred),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                            }
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=cv_scoring)
                        
                        # Display results
                        st.success(f"✅ {selected_model} trained successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Performance Metrics:**")
                            for metric, value in metrics.items():
                                st.metric(metric.upper(), f"{value:.4f}")
                        
                        with col2:
                            st.markdown("**Cross-Validation:**")
                            st.metric(f"CV {cv_scoring.upper()}", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                            st.info(f"**Training Time:** {training_time:.2f} seconds")
                            st.info(f"**Model:** {selected_model}")
                    
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        else:
            st.info("Clustering models don't require a target column. Feature configuration and training will be implemented for unsupervised learning.")
    
    # Configuration export/import
    st.markdown("---")
    st.markdown("**Configuration Management:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Configuration"):
            config = {
                'model_category': model_category,
                'selected_model': selected_model,
                'cv_folds': cv_folds,
                'cv_scoring': cv_scoring,
                'scaling_method': scaling_method,
                'feature_selection': feature_selection,
                'random_state': random_state,
                'timestamp': datetime.now().isoformat()
            }
            
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="Download Configuration",
                data=config_json,
                file_name=f"ml_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_config = st.file_uploader("📤 Import Configuration", type=['json'])
        
        if uploaded_config is not None:
            try:
                config = json.load(uploaded_config)
                st.success("Configuration loaded successfully!")
                st.json(config)
            except Exception as e:
                st.error(f"Failed to load configuration: {str(e)}")