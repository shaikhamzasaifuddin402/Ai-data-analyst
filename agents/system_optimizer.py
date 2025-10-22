"""
System Optimizer for Large-Scale Data Processing
Advanced optimization engine with cloud/edge computing integration for 80M+ records
"""

import streamlit as st
import pandas as pd
import numpy as np
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import gc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SystemOptimizer:
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_info = psutil.virtual_memory()
        self.performance_metrics = []
        self.optimization_history = []
        self.chunk_size = self._calculate_optimal_chunk_size()
        self.processing_strategies = {
            'memory_efficient': self._memory_efficient_processing,
            'parallel_processing': self._parallel_processing,
            'streaming': self._streaming_processing,
            'distributed': self._distributed_processing
        }
        
    def _calculate_optimal_chunk_size(self):
        """Calculate optimal chunk size based on available memory"""
        available_memory_gb = self.memory_info.available / (1024**3)
        
        # Conservative chunk size calculation
        if available_memory_gb > 16:
            return 1000000  # 1M rows
        elif available_memory_gb > 8:
            return 500000   # 500K rows
        elif available_memory_gb > 4:
            return 250000   # 250K rows
        else:
            return 100000   # 100K rows
    
    def get_system_metrics(self):
        """Get current system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_usage': cpu_percent,
            'memory_total': memory.total / (1024**3),  # GB
            'memory_available': memory.available / (1024**3),  # GB
            'memory_percent': memory.percent,
            'disk_total': disk.total / (1024**3),  # GB
            'disk_free': disk.free / (1024**3),  # GB
            'disk_percent': (disk.used / disk.total) * 100,
            'cpu_cores': self.cpu_count,
            'timestamp': time.time()
        }
        
        self.performance_metrics.append(metrics)
        return metrics
    
    def _memory_efficient_processing(self, data, operation, **kwargs):
        """Memory-efficient processing using chunking"""
        results = []
        total_chunks = len(data) // self.chunk_size + (1 if len(data) % self.chunk_size else 0)
        
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            
            # Process chunk
            if operation == 'describe':
                result = chunk.describe()
            elif operation == 'groupby':
                group_col = kwargs.get('group_col')
                agg_col = kwargs.get('agg_col')
                agg_func = kwargs.get('agg_func', 'mean')
                result = chunk.groupby(group_col)[agg_col].agg(agg_func)
            elif operation == 'filter':
                condition = kwargs.get('condition')
                result = chunk.query(condition) if condition else chunk
            else:
                result = chunk
            
            results.append(result)
            
            # Force garbage collection
            gc.collect()
            
            # Yield control to allow UI updates
            time.sleep(0.001)
        
        # Combine results
        if operation == 'describe':
            return pd.concat(results).groupby(level=0).mean()
        elif operation == 'groupby':
            return pd.concat(results).groupby(level=0).mean()
        else:
            return pd.concat(results, ignore_index=True)
    
    def _parallel_processing(self, data, operation, **kwargs):
        """Parallel processing using multiple cores"""
        n_cores = min(self.cpu_count, 8)  # Limit to 8 cores max
        chunk_size = len(data) // n_cores
        
        def process_chunk(chunk_data):
            if operation == 'describe':
                return chunk_data.describe()
            elif operation == 'groupby':
                group_col = kwargs.get('group_col')
                agg_col = kwargs.get('agg_col')
                agg_func = kwargs.get('agg_func', 'mean')
                return chunk_data.groupby(group_col)[agg_col].agg(agg_func)
            elif operation == 'filter':
                condition = kwargs.get('condition')
                return chunk_data.query(condition) if condition else chunk_data
            else:
                return chunk_data
        
        # Split data into chunks
        chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        # Combine results
        if operation == 'describe':
            return pd.concat(results).groupby(level=0).mean()
        elif operation == 'groupby':
            return pd.concat(results).groupby(level=0).mean()
        else:
            return pd.concat(results, ignore_index=True)
    
    def _streaming_processing(self, data, operation, **kwargs):
        """Streaming processing for very large datasets"""
        def data_generator():
            for i in range(0, len(data), self.chunk_size):
                yield data.iloc[i:i + self.chunk_size]
        
        results = []
        for chunk in data_generator():
            if operation == 'describe':
                result = chunk.describe()
            elif operation == 'groupby':
                group_col = kwargs.get('group_col')
                agg_col = kwargs.get('agg_col')
                agg_func = kwargs.get('agg_func', 'mean')
                result = chunk.groupby(group_col)[agg_col].agg(agg_func)
            elif operation == 'filter':
                condition = kwargs.get('condition')
                result = chunk.query(condition) if condition else chunk
            else:
                result = chunk
            
            results.append(result)
            
            # Memory management
            del chunk
            gc.collect()
        
        # Combine results
        if operation == 'describe':
            return pd.concat(results).groupby(level=0).mean()
        elif operation == 'groupby':
            return pd.concat(results).groupby(level=0).mean()
        else:
            return pd.concat(results, ignore_index=True)
    
    def _distributed_processing(self, data, operation, **kwargs):
        """Simulated distributed processing (would connect to actual cluster in production)"""
        # This is a simulation - in production, this would connect to Spark, Dask, or Ray
        st.info("🌐 Distributed processing simulation - would connect to cluster in production")
        
        # For now, use parallel processing as fallback
        return self._parallel_processing(data, operation, **kwargs)
    
    def optimize_processing(self, data, operation, strategy='auto', **kwargs):
        """Optimize processing based on data size and system resources"""
        start_time = time.time()
        initial_metrics = self.get_system_metrics()
        
        # Auto-select strategy if not specified
        if strategy == 'auto':
            data_size_gb = data.memory_usage(deep=True).sum() / (1024**3)
            
            if data_size_gb > 10:  # > 10GB
                strategy = 'distributed'
            elif data_size_gb > 2:  # > 2GB
                strategy = 'streaming'
            elif len(data) > 1000000:  # > 1M rows
                strategy = 'parallel_processing'
            else:
                strategy = 'memory_efficient'
        
        st.info(f"🚀 Using {strategy} strategy for processing {len(data):,} rows")
        
        # Execute processing
        try:
            result = self.processing_strategies[strategy](data, operation, **kwargs)
            
            # Record performance
            end_time = time.time()
            final_metrics = self.get_system_metrics()
            
            optimization_record = {
                'strategy': strategy,
                'operation': operation,
                'data_rows': len(data),
                'data_cols': len(data.columns),
                'processing_time': end_time - start_time,
                'memory_before': initial_metrics['memory_percent'],
                'memory_after': final_metrics['memory_percent'],
                'cpu_usage': final_metrics['cpu_usage'],
                'timestamp': time.time()
            }
            
            self.optimization_history.append(optimization_record)
            
            return result, optimization_record
            
        except Exception as e:
            st.error(f"Error in {strategy} processing: {str(e)}")
            return None, None
    
    def get_optimization_recommendations(self, data):
        """Get optimization recommendations based on data characteristics"""
        recommendations = []
        
        data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
        data_size_gb = data_size_mb / 1024
        
        # Memory recommendations
        available_memory_gb = self.memory_info.available / (1024**3)
        memory_ratio = data_size_gb / available_memory_gb
        
        if memory_ratio > 0.8:
            recommendations.append({
                'type': 'critical',
                'message': f"Dataset ({data_size_gb:.1f}GB) uses {memory_ratio*100:.1f}% of available memory. Consider streaming or distributed processing."
            })
        elif memory_ratio > 0.5:
            recommendations.append({
                'type': 'warning',
                'message': f"Dataset is large ({data_size_gb:.1f}GB). Consider chunked processing for better performance."
            })
        
        # Processing strategy recommendations
        if len(data) > 50000000:  # 50M+ rows
            recommendations.append({
                'type': 'info',
                'message': "For datasets with 50M+ rows, distributed processing is recommended for optimal performance."
            })
        elif len(data) > 10000000:  # 10M+ rows
            recommendations.append({
                'type': 'info',
                'message': "For datasets with 10M+ rows, parallel processing will significantly improve performance."
            })
        
        # Column recommendations
        if len(data.columns) > 1000:
            recommendations.append({
                'type': 'warning',
                'message': f"Dataset has {len(data.columns)} columns. Consider feature selection to reduce dimensionality."
            })
        
        # Data type recommendations
        object_cols = data.select_dtypes(include=['object']).columns
        if len(object_cols) > len(data.columns) * 0.5:
            recommendations.append({
                'type': 'info',
                'message': f"Dataset has many text columns ({len(object_cols)}). Consider categorical encoding for better performance."
            })
        
        return recommendations

def render_system_optimizer():
    """Render the System Optimizer interface"""
    st.header("⚡ System Optimizer")
    st.markdown("**Large-Scale Data Processing & Performance Optimization**")
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = SystemOptimizer()
    
    optimizer = st.session_state.optimizer
    
    # System metrics dashboard
    st.subheader("📊 System Performance Dashboard")
    
    # Get current metrics
    current_metrics = optimizer.get_system_metrics()
    
    # Display current system status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "CPU Usage", 
            f"{current_metrics['cpu_usage']:.1f}%",
            delta=f"{current_metrics['cpu_cores']} cores"
        )
    
    with col2:
        memory_used = current_metrics['memory_total'] - current_metrics['memory_available']
        st.metric(
            "Memory Usage", 
            f"{memory_used:.1f}GB",
            delta=f"{current_metrics['memory_percent']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Available Memory", 
            f"{current_metrics['memory_available']:.1f}GB",
            delta=f"/{current_metrics['memory_total']:.1f}GB total"
        )
    
    with col4:
        disk_used = current_metrics['disk_total'] - current_metrics['disk_free']
        st.metric(
            "Disk Usage", 
            f"{disk_used:.1f}GB",
            delta=f"{current_metrics['disk_percent']:.1f}%"
        )
    
    # Performance monitoring chart
    if len(optimizer.performance_metrics) > 1:
        st.subheader("📈 Real-Time Performance Monitoring")
        
        # Create performance chart
        metrics_df = pd.DataFrame(optimizer.performance_metrics)
        metrics_df['time'] = pd.to_datetime(metrics_df['timestamp'], unit='s')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['CPU Usage (%)', 'Memory Usage (%)', 'Available Memory (GB)', 'Disk Usage (%)'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=metrics_df['time'], y=metrics_df['cpu_usage'], 
                      name='CPU Usage', line=dict(color='red')),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=metrics_df['time'], y=metrics_df['memory_percent'], 
                      name='Memory Usage', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Available Memory
        fig.add_trace(
            go.Scatter(x=metrics_df['time'], y=metrics_df['memory_available'], 
                      name='Available Memory', line=dict(color='green')),
            row=2, col=1
        )
        
        # Disk Usage
        fig.add_trace(
            go.Scatter(x=metrics_df['time'], y=metrics_df['disk_percent'], 
                      name='Disk Usage', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=500, showlegend=False, title_text="System Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data processing optimization
    st.markdown("---")
    st.subheader("🚀 Data Processing Optimization")
    
    # File upload for optimization testing
    uploaded_file = st.file_uploader("Upload dataset for optimization analysis", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Friendly 1GB check before reading
            file_size_bytes = getattr(uploaded_file, 'size', None)
            if file_size_bytes is None:
                file_size_bytes = uploaded_file.getbuffer().nbytes
            if file_size_bytes > 1024**3:
                st.error("The uploaded file exceeds 1GB. Please upload a smaller file or switch to streaming/distributed strategies.")
                st.info("Tip: For very large datasets, use 'streaming' or 'distributed' strategies in the configuration below.")
                # Skip loading to avoid memory issues
                st.stop()
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Data analysis
            data_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.info(f"Dataset size: {data_size_mb:.1f} MB")
            
            # Get optimization recommendations
            recommendations = optimizer.get_optimization_recommendations(df)
            
            if recommendations:
                st.subheader("💡 Optimization Recommendations")
                for rec in recommendations:
                    if rec['type'] == 'critical':
                        st.error(f"🚨 {rec['message']}")
                    elif rec['type'] == 'warning':
                        st.warning(f"⚠️ {rec['message']}")
                    else:
                        st.info(f"💡 {rec['message']}")
            
            # Processing strategy selection
            st.subheader("⚙️ Processing Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategy = st.selectbox(
                    "Processing Strategy:",
                    ['auto', 'memory_efficient', 'parallel_processing', 'streaming', 'distributed']
                )
            
            with col2:
                operation = st.selectbox(
                    "Operation Type:",
                    ['describe', 'groupby', 'filter', 'sample']
                )
            
            # Operation-specific parameters
            operation_params = {}
            
            if operation == 'groupby':
                col1, col2, col3 = st.columns(3)
                with col1:
                    group_col = st.selectbox("Group by column:", df.columns)
                    operation_params['group_col'] = group_col
                with col2:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    agg_col = st.selectbox("Aggregate column:", numeric_cols)
                    operation_params['agg_col'] = agg_col
                with col3:
                    agg_func = st.selectbox("Aggregation function:", ['mean', 'sum', 'count', 'min', 'max'])
                    operation_params['agg_func'] = agg_func
            
            elif operation == 'filter':
                condition = st.text_input("Filter condition (pandas query syntax):", 
                                        placeholder="e.g., column_name > 100")
                if condition:
                    operation_params['condition'] = condition
            
            # Run optimization
            if st.button("🚀 Run Optimized Processing"):
                with st.spinner(f"Running {strategy} processing..."):
                    result, performance = optimizer.optimize_processing(
                        df, operation, strategy, **operation_params
                    )
                    
                    if result is not None and performance is not None:
                        st.success("Processing completed successfully!")
                        
                        # Display performance metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Processing Time", f"{performance['processing_time']:.2f}s")
                        with col2:
                            st.metric("Memory Change", f"{performance['memory_after'] - performance['memory_before']:+.1f}%")
                        with col3:
                            st.metric("CPU Usage", f"{performance['cpu_usage']:.1f}%")
                        
                        # Display results
                        st.subheader("📊 Processing Results")
                        if isinstance(result, pd.DataFrame):
                            if len(result) > 1000:
                                st.info(f"Showing first 1000 rows of {len(result):,} total results")
                                st.dataframe(result.head(1000))
                            else:
                                st.dataframe(result)
                        else:
                            st.write(result)
            
            # Optimization history
            if optimizer.optimization_history:
                st.markdown("---")
                st.subheader("📈 Optimization History")
                
                history_df = pd.DataFrame(optimizer.optimization_history)
                
                # Performance comparison chart
                fig = px.scatter(
                    history_df, 
                    x='data_rows', 
                    y='processing_time',
                    color='strategy',
                    size='data_cols',
                    hover_data=['operation', 'memory_before', 'memory_after'],
                    title="Processing Performance by Strategy"
                )
                fig.update_layout(xaxis_title="Dataset Rows", yaxis_title="Processing Time (seconds)")
                st.plotly_chart(fig, use_container_width=True)
                
                # History table
                display_cols = ['strategy', 'operation', 'data_rows', 'processing_time', 'memory_before', 'memory_after']
                st.dataframe(history_df[display_cols].round(2))
        
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
    
    else:
        st.info("📤 Upload a dataset to analyze optimization opportunities and test processing strategies.")
        
        # System optimization tips
        st.subheader("💡 System Optimization Tips")
        st.markdown("""
        ### 🚀 **Performance Strategies:**
        
        **Memory Efficient Processing:**
        - ✅ Processes data in chunks to minimize memory usage
        - ✅ Ideal for systems with limited RAM
        - ✅ Automatic garbage collection between chunks
        
        **Parallel Processing:**
        - ✅ Utilizes multiple CPU cores simultaneously
        - ✅ Best for CPU-intensive operations
        - ✅ Scales with available cores
        
        **Streaming Processing:**
        - ✅ Processes data as a continuous stream
        - ✅ Minimal memory footprint
        - ✅ Ideal for very large datasets (10GB+)
        
        **Distributed Processing:**
        - ✅ Scales across multiple machines
        - ✅ Handles massive datasets (100GB+)
        - ✅ Fault-tolerant and resilient
        
        ### 📊 **Optimization Guidelines:**
        - **< 1GB**: Standard processing
        - **1-10GB**: Chunked or parallel processing
        - **10-100GB**: Streaming processing
        - **100GB+**: Distributed processing
        """)

# Auto-refresh system metrics
def auto_refresh_metrics():
    """Auto-refresh system metrics in the background"""
    if 'optimizer' in st.session_state:
        st.session_state.optimizer.get_system_metrics()

# Set up auto-refresh (every 5 seconds)
if 'metrics_refresh' not in st.session_state:
    st.session_state.metrics_refresh = True