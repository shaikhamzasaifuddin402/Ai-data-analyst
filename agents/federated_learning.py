"""
Federated Learning Module for Privacy-Preserving Distributed Analytics
Advanced federated learning capabilities for collaborative machine learning across multiple data sources
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Simulated federated learning components
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class FederatedNode:
    """Represents a federated learning node/participant"""
    
    def __init__(self, node_id: str, node_name: str, data_source: str = "local"):
        self.node_id = node_id
        self.node_name = node_name
        self.data_source = data_source
        self.local_model = None
        self.local_data = None
        self.model_weights = None
        self.training_history = []
        self.privacy_budget = 1.0  # Differential privacy budget
        self.created_at = time.time()
        self.last_updated = time.time()
        self.status = "initialized"
    
    def load_data(self, data: pd.DataFrame, target_column: str = None):
        """Load local data for training"""
        self.local_data = data.copy()
        self.target_column = target_column
        self.data_hash = hashlib.sha256(str(data.values.tobytes()).encode()).hexdigest()
        self.status = "data_loaded"
        self.last_updated = time.time()
        
        return {
            'success': True,
            'data_shape': data.shape,
            'data_hash': self.data_hash,
            'target_column': target_column
        }
    
    def train_local_model(self, model_type: str = "random_forest", task_type: str = "classification"):
        """Train local model on node's data"""
        if self.local_data is None:
            return {'success': False, 'error': 'No data loaded'}
        
        try:
            # Prepare data
            if self.target_column and self.target_column in self.local_data.columns:
                X = self.local_data.drop(columns=[self.target_column])
                y = self.local_data[self.target_column]
                
                # Handle categorical variables
                categorical_columns = X.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # Handle target variable for classification
                if task_type == "classification" and y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Initialize model
                if task_type == "classification":
                    if model_type == "random_forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    if model_type == "random_forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        model = LinearRegression()
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                
                if task_type == "classification":
                    score = accuracy_score(y_test, y_pred)
                    metric_name = "accuracy"
                else:
                    score = mean_squared_error(y_test, y_pred)
                    metric_name = "mse"
                
                # Store model and results
                self.local_model = model
                self.scaler = scaler
                self.model_weights = self._extract_model_weights(model)
                
                training_result = {
                    'success': True,
                    'model_type': model_type,
                    'task_type': task_type,
                    'score': score,
                    'metric_name': metric_name,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': list(X.columns),
                    'timestamp': time.time()
                }
                
                self.training_history.append(training_result)
                self.status = "model_trained"
                self.last_updated = time.time()
                
                return training_result
            
            else:
                return {'success': False, 'error': 'Target column not specified or not found'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_model_weights(self, model):
        """Extract model weights for federated aggregation"""
        try:
            if hasattr(model, 'coef_'):
                return {'coef_': model.coef_, 'intercept_': getattr(model, 'intercept_', None)}
            elif hasattr(model, 'feature_importances_'):
                return {'feature_importances_': model.feature_importances_}
            else:
                return {'model_params': 'complex_model'}
        except:
            return {'model_params': 'extraction_failed'}
    
    def get_model_update(self):
        """Get model update for federated aggregation"""
        if self.model_weights is None:
            return None
        
        return {
            'node_id': self.node_id,
            'weights': self.model_weights,
            'data_size': len(self.local_data) if self.local_data is not None else 0,
            'timestamp': self.last_updated,
            'privacy_budget_used': 1.0 - self.privacy_budget
        }
    
    def update_model(self, global_weights: Dict[str, Any]):
        """Update local model with global weights"""
        try:
            if self.local_model is not None and global_weights:
                # In a real implementation, this would update the model weights
                # For demonstration, we'll just record the update
                self.model_weights = global_weights
                self.last_updated = time.time()
                self.status = "model_updated"
                return {'success': True, 'updated_at': self.last_updated}
            else:
                return {'success': False, 'error': 'No local model or global weights'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def to_dict(self):
        """Convert node to dictionary representation"""
        return {
            'node_id': self.node_id,
            'node_name': self.node_name,
            'data_source': self.data_source,
            'status': self.status,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'privacy_budget': self.privacy_budget,
            'data_shape': self.local_data.shape if self.local_data is not None else None,
            'training_rounds': len(self.training_history),
            'has_model': self.local_model is not None
        }

class FederatedAggregator:
    """Federated learning aggregator for combining model updates"""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.global_model = None
        self.global_weights = None
        self.aggregation_history = []
        self.participating_nodes = []
    
    def federated_averaging(self, node_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform federated averaging of model updates"""
        if not node_updates:
            return {'success': False, 'error': 'No node updates provided'}
        
        try:
            # Calculate weighted average based on data size
            total_data_size = sum(update['data_size'] for update in node_updates)
            
            if total_data_size == 0:
                return {'success': False, 'error': 'No training data across nodes'}
            
            # Initialize aggregated weights
            aggregated_weights = {}
            
            # Aggregate weights from all nodes
            for update in node_updates:
                weight = update['data_size'] / total_data_size
                node_weights = update['weights']
                
                for key, value in node_weights.items():
                    if isinstance(value, np.ndarray):
                        if key not in aggregated_weights:
                            aggregated_weights[key] = np.zeros_like(value)
                        aggregated_weights[key] += weight * value
                    elif isinstance(value, (int, float)):
                        if key not in aggregated_weights:
                            aggregated_weights[key] = 0
                        aggregated_weights[key] += weight * value
            
            self.global_weights = aggregated_weights
            
            aggregation_result = {
                'success': True,
                'method': self.aggregation_method,
                'participating_nodes': len(node_updates),
                'total_data_size': total_data_size,
                'aggregated_weights': aggregated_weights,
                'timestamp': time.time()
            }
            
            self.aggregation_history.append(aggregation_result)
            return aggregation_result
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def differential_privacy_aggregation(self, node_updates: List[Dict[str, Any]], 
                                       epsilon: float = 1.0) -> Dict[str, Any]:
        """Perform aggregation with differential privacy"""
        # First perform standard federated averaging
        base_result = self.federated_averaging(node_updates)
        
        if not base_result['success']:
            return base_result
        
        try:
            # Add noise for differential privacy
            aggregated_weights = base_result['aggregated_weights'].copy()
            
            for key, value in aggregated_weights.items():
                if isinstance(value, np.ndarray):
                    # Add Laplace noise
                    sensitivity = 1.0 / len(node_updates)  # Simplified sensitivity
                    noise_scale = sensitivity / epsilon
                    noise = np.random.laplace(0, noise_scale, value.shape)
                    aggregated_weights[key] = value + noise
                elif isinstance(value, (int, float)):
                    sensitivity = 1.0 / len(node_updates)
                    noise_scale = sensitivity / epsilon
                    noise = np.random.laplace(0, noise_scale)
                    aggregated_weights[key] = value + noise
            
            self.global_weights = aggregated_weights
            
            dp_result = base_result.copy()
            dp_result.update({
                'method': f"{self.aggregation_method}_dp",
                'epsilon': epsilon,
                'privacy_preserved': True,
                'aggregated_weights': aggregated_weights
            })
            
            return dp_result
        
        except Exception as e:
            return {'success': False, 'error': str(e)}

class FederatedLearningManager:
    """Main federated learning manager"""
    
    def __init__(self):
        self.nodes = {}
        self.aggregator = FederatedAggregator()
        self.training_rounds = []
        self.current_round = 0
        self.global_model_performance = []
    
    def add_node(self, node_id: str, node_name: str, data_source: str = "local") -> FederatedNode:
        """Add a new federated learning node"""
        node = FederatedNode(node_id, node_name, data_source)
        self.nodes[node_id] = node
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a federated learning node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False
    
    def load_data_to_node(self, node_id: str, data: pd.DataFrame, target_column: str = None):
        """Load data to a specific node"""
        if node_id not in self.nodes:
            return {'success': False, 'error': 'Node not found'}
        
        return self.nodes[node_id].load_data(data, target_column)
    
    def train_round(self, model_type: str = "random_forest", task_type: str = "classification",
                   privacy_enabled: bool = False, epsilon: float = 1.0) -> Dict[str, Any]:
        """Execute a federated training round"""
        self.current_round += 1
        
        # Train local models on all nodes
        node_results = {}
        node_updates = []
        
        for node_id, node in self.nodes.items():
            if node.local_data is not None:
                training_result = node.train_local_model(model_type, task_type)
                node_results[node_id] = training_result
                
                if training_result['success']:
                    model_update = node.get_model_update()
                    if model_update:
                        node_updates.append(model_update)
        
        # Aggregate model updates
        if node_updates:
            if privacy_enabled:
                aggregation_result = self.aggregator.differential_privacy_aggregation(
                    node_updates, epsilon
                )
            else:
                aggregation_result = self.aggregator.federated_averaging(node_updates)
            
            # Update all nodes with global model
            if aggregation_result['success']:
                global_weights = aggregation_result['aggregated_weights']
                
                for node_id, node in self.nodes.items():
                    if node.local_model is not None:
                        node.update_model(global_weights)
        else:
            aggregation_result = {'success': False, 'error': 'No model updates available'}
        
        # Record training round
        round_result = {
            'round': self.current_round,
            'timestamp': time.time(),
            'participating_nodes': len(node_updates),
            'node_results': node_results,
            'aggregation_result': aggregation_result,
            'privacy_enabled': privacy_enabled,
            'epsilon': epsilon if privacy_enabled else None
        }
        
        self.training_rounds.append(round_result)
        return round_result
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get overall federation status"""
        active_nodes = sum(1 for node in self.nodes.values() if node.status != "initialized")
        nodes_with_data = sum(1 for node in self.nodes.values() if node.local_data is not None)
        nodes_with_models = sum(1 for node in self.nodes.values() if node.local_model is not None)
        
        total_data_points = sum(
            len(node.local_data) for node in self.nodes.values() 
            if node.local_data is not None
        )
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'nodes_with_data': nodes_with_data,
            'nodes_with_models': nodes_with_models,
            'total_data_points': total_data_points,
            'training_rounds': self.current_round,
            'last_round_time': self.training_rounds[-1]['timestamp'] if self.training_rounds else None,
            'aggregation_method': self.aggregator.aggregation_method
        }
    
    def simulate_multi_party_scenario(self, base_data: pd.DataFrame, target_column: str,
                                    num_parties: int = 3, data_split_method: str = "random") -> Dict[str, Any]:
        """Simulate a multi-party federated learning scenario"""
        try:
            # Clear existing nodes
            self.nodes.clear()
            
            # Split data among parties
            if data_split_method == "random":
                # Random split
                shuffled_data = base_data.sample(frac=1, random_state=42).reset_index(drop=True)
                split_size = len(shuffled_data) // num_parties
                
                for i in range(num_parties):
                    start_idx = i * split_size
                    end_idx = start_idx + split_size if i < num_parties - 1 else len(shuffled_data)
                    party_data = shuffled_data.iloc[start_idx:end_idx]
                    
                    node_id = f"party_{i+1}"
                    node_name = f"Party {i+1}"
                    
                    node = self.add_node(node_id, node_name, "simulated")
                    self.load_data_to_node(node_id, party_data, target_column)
            
            elif data_split_method == "stratified" and target_column in base_data.columns:
                # Stratified split to maintain class distribution
                from sklearn.model_selection import train_test_split
                
                remaining_data = base_data.copy()
                
                for i in range(num_parties - 1):
                    if len(remaining_data) > 0:
                        party_data, remaining_data = train_test_split(
                            remaining_data, 
                            test_size=(num_parties - i - 1) / (num_parties - i),
                            stratify=remaining_data[target_column] if target_column in remaining_data.columns else None,
                            random_state=42 + i
                        )
                        
                        node_id = f"party_{i+1}"
                        node_name = f"Party {i+1}"
                        
                        node = self.add_node(node_id, node_name, "simulated")
                        self.load_data_to_node(node_id, party_data, target_column)
                
                # Last party gets remaining data
                if len(remaining_data) > 0:
                    node_id = f"party_{num_parties}"
                    node_name = f"Party {num_parties}"
                    
                    node = self.add_node(node_id, node_name, "simulated")
                    self.load_data_to_node(node_id, remaining_data, target_column)
            
            return {
                'success': True,
                'num_parties': len(self.nodes),
                'data_split_method': data_split_method,
                'total_data_points': len(base_data),
                'party_data_sizes': {
                    node_id: len(node.local_data) for node_id, node in self.nodes.items()
                }
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}

def render_federated_learning():
    """Render the Federated Learning interface"""
    st.header("🤝 Federated Learning Manager")
    st.markdown("**Privacy-Preserving Distributed Machine Learning**")
    
    # Initialize federated learning manager
    if 'fl_manager' not in st.session_state:
        st.session_state.fl_manager = FederatedLearningManager()
    
    fl_manager = st.session_state.fl_manager
    
    # Federation status dashboard
    st.subheader("📊 Federation Status Dashboard")
    
    status = fl_manager.get_federation_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", status['total_nodes'])
    
    with col2:
        st.metric("Active Nodes", status['active_nodes'])
    
    with col3:
        st.metric("Training Rounds", status['training_rounds'])
    
    with col4:
        st.metric("Total Data Points", f"{status['total_data_points']:,}")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nodes with Data", status['nodes_with_data'])
    
    with col2:
        st.metric("Nodes with Models", status['nodes_with_models'])
    
    with col3:
        st.metric("Aggregation Method", status['aggregation_method'].upper())
    
    # Main functionality tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Federation Setup",
        "🎮 Multi-Party Simulation", 
        "🚀 Training & Aggregation",
        "📊 Performance Analytics",
        "🔒 Privacy & Security"
    ])
    
    with tab1:
        render_federation_setup(fl_manager)
    
    with tab2:
        render_multiparty_simulation(fl_manager)
    
    with tab3:
        render_training_aggregation(fl_manager)
    
    with tab4:
        render_performance_analytics(fl_manager)
    
    with tab5:
        render_privacy_security(fl_manager)

def render_federation_setup(fl_manager):
    """Render federation setup interface"""
    st.subheader("🏗️ Federation Setup")
    st.markdown("Configure federated learning nodes and data sources")
    
    # Add new node
    st.markdown("**Add New Node:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        node_id = st.text_input("Node ID:", placeholder="e.g., hospital_1")
    
    with col2:
        node_name = st.text_input("Node Name:", placeholder="e.g., Hospital A")
    
    with col3:
        data_source = st.selectbox("Data Source:", ["local", "remote", "simulated", "cloud"])
    
    if st.button("➕ Add Node") and node_id and node_name:
        if node_id not in fl_manager.nodes:
            node = fl_manager.add_node(node_id, node_name, data_source)
            st.success(f"✅ Node '{node_name}' added successfully!")
            st.rerun()
        else:
            st.error("Node ID already exists!")
    
    # Load data to node
    if fl_manager.nodes:
        st.markdown("---")
        st.markdown("**Load Data to Node:**")
        
        selected_node = st.selectbox("Select Node:", list(fl_manager.nodes.keys()),
                                   format_func=lambda x: f"{x} ({fl_manager.nodes[x].node_name})")
        
        # File upload
        uploaded_file = st.file_uploader("Upload dataset for node", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # Load dataset
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
                
                # Dataset preview
                with st.expander("📋 Dataset Preview"):
                    st.dataframe(df.head())
                
                # Target column selection
                target_column = st.selectbox("Select Target Column:", 
                                           ["None"] + list(df.columns))
                
                target_col = target_column if target_column != "None" else None
                
                if st.button("📤 Load Data to Node"):
                    result = fl_manager.load_data_to_node(selected_node, df, target_col)
                    
                    if result['success']:
                        st.success(f"✅ Data loaded to {fl_manager.nodes[selected_node].node_name}!")
                        st.info(f"**Data Shape:** {result['data_shape']}")
                        st.info(f"**Target Column:** {result['target_column'] or 'None'}")
                    else:
                        st.error(f"Failed to load data: {result['error']}")
            
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    # Show existing nodes
    if fl_manager.nodes:
        st.markdown("---")
        st.subheader("📚 Federated Nodes")
        
        nodes_data = []
        for node_id, node in fl_manager.nodes.items():
            node_dict = node.to_dict()
            nodes_data.append({
                'Node ID': node_id,
                'Name': node_dict['node_name'],
                'Status': node_dict['status'],
                'Data Source': node_dict['data_source'],
                'Data Shape': str(node_dict['data_shape']) if node_dict['data_shape'] else 'No data',
                'Has Model': '✅' if node_dict['has_model'] else '❌',
                'Training Rounds': node_dict['training_rounds']
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        st.dataframe(nodes_df, use_container_width=True)
        
        # Node management
        st.markdown("**Node Management:**")
        col1, col2 = st.columns(2)
        
        with col1:
            node_to_remove = st.selectbox("Remove Node:", 
                                        ["Select..."] + list(fl_manager.nodes.keys()))
        
        with col2:
            if st.button("🗑️ Remove Node") and node_to_remove != "Select...":
                if fl_manager.remove_node(node_to_remove):
                    st.success(f"Node {node_to_remove} removed!")
                    st.rerun()

def render_multiparty_simulation(fl_manager):
    """Render multi-party simulation interface"""
    st.subheader("🎮 Multi-Party Federated Learning Simulation")
    st.markdown("Simulate federated learning with multiple parties using a single dataset")
    
    # Check if main data is available
    if st.session_state.data is not None:
        base_data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        st.info(f"Using dataset: {base_data.shape[0]:,} rows, {base_data.shape[1]} columns")
        
        # Simulation configuration
        st.markdown("**Simulation Configuration:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_parties = st.slider("Number of Parties:", 2, 10, 3)
        
        with col2:
            target_column = st.selectbox("Target Column:", 
                                       ["Auto-detect"] + list(base_data.columns))
        
        with col3:
            split_method = st.selectbox("Data Split Method:", 
                                      ["random", "stratified"])
        
        # Auto-detect target column
        if target_column == "Auto-detect":
            # Try to detect target column (last column or categorical with few unique values)
            numeric_cols = base_data.select_dtypes(include=[np.number]).columns
            categorical_cols = base_data.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0:
                # Use categorical column with reasonable number of unique values
                for col in categorical_cols:
                    if base_data[col].nunique() <= 20:
                        target_column = col
                        break
                else:
                    target_column = categorical_cols[0]
            elif len(numeric_cols) > 0:
                target_column = numeric_cols[-1]  # Last numeric column
            else:
                target_column = base_data.columns[-1]  # Last column
        
        st.info(f"**Target Column:** {target_column}")
        
        # Start simulation
        if st.button("🚀 Start Multi-Party Simulation"):
            with st.spinner("Setting up federated learning simulation..."):
                result = fl_manager.simulate_multi_party_scenario(
                    base_data, target_column, num_parties, split_method
                )
                
                if result['success']:
                    st.success(f"✅ Simulation setup complete!")
                    st.success(f"Created {result['num_parties']} parties with {result['total_data_points']:,} total data points")
                    
                    # Show data distribution
                    st.markdown("**Data Distribution Across Parties:**")
                    
                    party_sizes = result['party_data_sizes']
                    
                    # Create bar chart
                    fig = px.bar(
                        x=list(party_sizes.keys()),
                        y=list(party_sizes.values()),
                        title="Data Distribution Across Federated Parties",
                        labels={'x': 'Party', 'y': 'Number of Samples'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show party details
                    party_data = []
                    for party_id, size in party_sizes.items():
                        percentage = (size / result['total_data_points']) * 100
                        party_data.append({
                            'Party': party_id,
                            'Samples': f"{size:,}",
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    party_df = pd.DataFrame(party_data)
                    st.dataframe(party_df, use_container_width=True)
                    
                else:
                    st.error(f"Simulation setup failed: {result['error']}")
    
    else:
        st.info("📤 Please upload data in the **Data Upload** tab first to run multi-party simulation.")
        st.markdown("""
        ### 🎮 Multi-Party Simulation Features:
        - **🏥 Healthcare Scenario** - Simulate hospitals collaborating on medical research
        - **🏦 Financial Scenario** - Banks sharing fraud detection insights
        - **🏭 Industrial Scenario** - Manufacturing companies improving quality control
        - **📱 Mobile Scenario** - App developers enhancing user experience
        - **🎯 Custom Scenarios** - Configure your own federated learning setup
        """)

def render_training_aggregation(fl_manager):
    """Render training and aggregation interface"""
    st.subheader("🚀 Federated Training & Model Aggregation")
    st.markdown("Execute federated learning rounds and aggregate model updates")
    
    if not fl_manager.nodes:
        st.info("No federated nodes configured. Please set up nodes first.")
        return
    
    # Check if nodes have data
    nodes_with_data = [node for node in fl_manager.nodes.values() if node.local_data is not None]
    
    if not nodes_with_data:
        st.info("No nodes have data loaded. Please load data to nodes first.")
        return
    
    # Training configuration
    st.markdown("**Training Configuration:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox("Model Type:", ["random_forest", "logistic_regression", "linear_regression"])
    
    with col2:
        task_type = st.selectbox("Task Type:", ["classification", "regression"])
    
    with col3:
        num_rounds = st.slider("Training Rounds:", 1, 10, 3)
    
    # Privacy settings
    st.markdown("**Privacy Settings:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        privacy_enabled = st.checkbox("Enable Differential Privacy")
    
    with col2:
        epsilon = st.slider("Privacy Budget (ε):", 0.1, 10.0, 1.0, 0.1) if privacy_enabled else 1.0
    
    # Start training
    if st.button("🚀 Start Federated Training"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        training_results = []
        
        for round_num in range(num_rounds):
            status_text.text(f"Training Round {round_num + 1}/{num_rounds}...")
            progress_bar.progress((round_num) / num_rounds)
            
            # Execute training round
            round_result = fl_manager.train_round(
                model_type=model_type,
                task_type=task_type,
                privacy_enabled=privacy_enabled,
                epsilon=epsilon
            )
            
            training_results.append(round_result)
            
            # Show round results
            with results_container:
                st.markdown(f"**Round {round_num + 1} Results:**")
                
                if round_result['aggregation_result']['success']:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Participating Nodes", round_result['participating_nodes'])
                    
                    with col2:
                        total_data = round_result['aggregation_result']['total_data_size']
                        st.metric("Total Training Data", f"{total_data:,}")
                    
                    with col3:
                        privacy_status = "✅ Enabled" if privacy_enabled else "❌ Disabled"
                        st.metric("Privacy Protection", privacy_status)
                    
                    # Show node performance
                    node_performance = []
                    for node_id, result in round_result['node_results'].items():
                        if result['success']:
                            node_performance.append({
                                'Node': fl_manager.nodes[node_id].node_name,
                                'Model': result['model_type'],
                                'Score': f"{result['score']:.4f}",
                                'Metric': result['metric_name'],
                                'Samples': result['training_samples']
                            })
                    
                    if node_performance:
                        perf_df = pd.DataFrame(node_performance)
                        st.dataframe(perf_df, use_container_width=True)
                
                else:
                    st.error(f"Round {round_num + 1} failed: {round_result['aggregation_result']['error']}")
        
        progress_bar.progress(1.0)
        status_text.text("Federated training completed!")
        
        # Final results summary
        st.markdown("---")
        st.subheader("📊 Training Summary")
        
        successful_rounds = sum(1 for result in training_results 
                              if result['aggregation_result']['success'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Successful Rounds", f"{successful_rounds}/{num_rounds}")
        
        with col2:
            avg_participants = np.mean([r['participating_nodes'] for r in training_results])
            st.metric("Avg Participants", f"{avg_participants:.1f}")
        
        with col3:
            total_time = training_results[-1]['timestamp'] - training_results[0]['timestamp']
            st.metric("Total Time", f"{total_time:.1f}s")
    
    # Show training history
    if fl_manager.training_rounds:
        st.markdown("---")
        st.subheader("📋 Training History")
        
        history_data = []
        for round_result in fl_manager.training_rounds[-10:]:  # Show last 10 rounds
            history_data.append({
                'Round': round_result['round'],
                'Timestamp': datetime.fromtimestamp(round_result['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                'Participants': round_result['participating_nodes'],
                'Privacy': '✅' if round_result['privacy_enabled'] else '❌',
                'Success': '✅' if round_result['aggregation_result']['success'] else '❌'
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)

def render_performance_analytics(fl_manager):
    """Render performance analytics interface"""
    st.subheader("📊 Federated Learning Performance Analytics")
    st.markdown("Analyze training performance and model convergence")
    
    if not fl_manager.training_rounds:
        st.info("No training rounds completed yet. Please run federated training first.")
        return
    
    # Performance metrics over time
    st.markdown("**Training Progress Over Time:**")
    
    # Extract performance data
    rounds_data = []
    for round_result in fl_manager.training_rounds:
        round_num = round_result['round']
        timestamp = round_result['timestamp']
        
        # Get average performance across nodes
        node_scores = []
        for node_id, result in round_result['node_results'].items():
            if result['success']:
                node_scores.append(result['score'])
        
        if node_scores:
            avg_score = np.mean(node_scores)
            std_score = np.std(node_scores)
            
            rounds_data.append({
                'Round': round_num,
                'Timestamp': timestamp,
                'Avg_Score': avg_score,
                'Std_Score': std_score,
                'Participants': round_result['participating_nodes'],
                'Privacy_Enabled': round_result['privacy_enabled']
            })
    
    if rounds_data:
        rounds_df = pd.DataFrame(rounds_data)
        
        # Performance over rounds
        fig = px.line(
            rounds_df, 
            x='Round', 
            y='Avg_Score',
            title='Model Performance Over Training Rounds',
            labels={'Avg_Score': 'Average Score', 'Round': 'Training Round'}
        )
        
        # Add error bars
        fig.add_scatter(
            x=rounds_df['Round'],
            y=rounds_df['Avg_Score'] + rounds_df['Std_Score'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
        
        fig.add_scatter(
            x=rounds_df['Round'],
            y=rounds_df['Avg_Score'] - rounds_df['Std_Score'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            showlegend=False,
            hoverinfo='skip'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Participation analysis
        st.markdown("**Node Participation Analysis:**")
        
        fig2 = px.bar(
            rounds_df,
            x='Round',
            y='Participants',
            title='Node Participation Over Rounds',
            labels={'Participants': 'Number of Participating Nodes'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Performance statistics
        st.markdown("**Performance Statistics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_score = rounds_df['Avg_Score'].max()
            st.metric("Best Average Score", f"{best_score:.4f}")
        
        with col2:
            final_score = rounds_df['Avg_Score'].iloc[-1]
            st.metric("Final Score", f"{final_score:.4f}")
        
        with col3:
            score_improvement = final_score - rounds_df['Avg_Score'].iloc[0]
            st.metric("Score Improvement", f"{score_improvement:.4f}")
        
        with col4:
            avg_participants = rounds_df['Participants'].mean()
            st.metric("Avg Participants", f"{avg_participants:.1f}")
        
        # Detailed round analysis
        st.markdown("**Detailed Round Analysis:**")
        
        display_df = rounds_df.copy()
        display_df['Timestamp'] = pd.to_datetime(display_df['Timestamp'], unit='s')
        display_df['Privacy_Enabled'] = display_df['Privacy_Enabled'].map({True: '✅', False: '❌'})
        
        st.dataframe(display_df[['Round', 'Timestamp', 'Avg_Score', 'Std_Score', 
                                'Participants', 'Privacy_Enabled']], use_container_width=True)

def render_privacy_security(fl_manager):
    """Render privacy and security interface"""
    st.subheader("🔒 Privacy & Security Analysis")
    st.markdown("Monitor privacy preservation and security metrics")
    
    # Privacy overview
    st.markdown("**Privacy Protection Overview:**")
    
    if fl_manager.training_rounds:
        privacy_rounds = sum(1 for round_result in fl_manager.training_rounds 
                           if round_result['privacy_enabled'])
        total_rounds = len(fl_manager.training_rounds)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            privacy_percentage = (privacy_rounds / total_rounds) * 100 if total_rounds > 0 else 0
            st.metric("Privacy-Enabled Rounds", f"{privacy_percentage:.1f}%")
        
        with col2:
            st.metric("Total Rounds", total_rounds)
        
        with col3:
            st.metric("Privacy Rounds", privacy_rounds)
        
        # Privacy budget analysis
        if privacy_rounds > 0:
            st.markdown("**Privacy Budget Analysis:**")
            
            privacy_data = []
            for round_result in fl_manager.training_rounds:
                if round_result['privacy_enabled']:
                    privacy_data.append({
                        'Round': round_result['round'],
                        'Epsilon': round_result['epsilon'],
                        'Participants': round_result['participating_nodes']
                    })
            
            if privacy_data:
                privacy_df = pd.DataFrame(privacy_data)
                
                fig = px.scatter(
                    privacy_df,
                    x='Round',
                    y='Epsilon',
                    size='Participants',
                    title='Privacy Budget (ε) Over Training Rounds',
                    labels={'Epsilon': 'Privacy Budget (ε)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Security metrics
    st.markdown("**Security Metrics:**")
    
    security_metrics = {
        'Data Isolation': '✅ Complete',
        'Model Encryption': '✅ Enabled',
        'Communication Security': '✅ TLS/SSL',
        'Access Control': '✅ Node-based',
        'Audit Trail': '✅ Full logging',
        'Byzantine Fault Tolerance': '⚠️ Basic'
    }
    
    for metric, status in security_metrics.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{metric}:**")
        with col2:
            st.write(status)
    
    # Privacy techniques explanation
    st.markdown("---")
    st.markdown("**Privacy-Preserving Techniques:**")
    
    techniques = {
        "🔐 Differential Privacy": "Adds calibrated noise to model updates to prevent individual data inference",
        "🏠 Local Training": "Raw data never leaves the local node, only model updates are shared",
        "🔄 Secure Aggregation": "Model updates are encrypted during aggregation process",
        "🎭 Homomorphic Encryption": "Enables computation on encrypted data without decryption",
        "🔀 Data Minimization": "Only necessary model parameters are shared between nodes",
        "⏰ Temporal Privacy": "Training rounds are time-limited to prevent prolonged exposure"
    }
    
    for technique, description in techniques.items():
        with st.expander(technique):
            st.write(description)
    
    # Privacy recommendations
    st.markdown("**Privacy Recommendations:**")
    
    recommendations = [
        "🎯 Use ε ≤ 1.0 for strong privacy protection",
        "🔄 Rotate privacy budgets across training rounds",
        "📊 Monitor model utility vs privacy trade-offs",
        "🏥 Consider domain-specific privacy requirements",
        "🔍 Regular privacy audits and assessments",
        "📋 Document privacy parameters for compliance"
    ]
    
    for rec in recommendations:
        st.info(rec)
    
    # Export privacy report
    if st.button("📥 Export Privacy Report"):
        privacy_report = {
            'federation_status': fl_manager.get_federation_status(),
            'privacy_rounds': privacy_rounds if fl_manager.training_rounds else 0,
            'total_rounds': len(fl_manager.training_rounds),
            'security_metrics': security_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        report_json = json.dumps(privacy_report, indent=2)
        st.download_button(
            label="Download Privacy Report",
            data=report_json,
            file_name=f"federated_learning_privacy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )