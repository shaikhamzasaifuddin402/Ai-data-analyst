"""
Dynamic Dashboard Module for Generative AI-Powered Data Analytics Platform

This module provides interactive visualization components with real-time updates,
customizable layouts, and advanced analytics capabilities.

Features:
- Interactive charts with drill-down capabilities
- Real-time data updates and auto-refresh
- Customizable dashboard layouts
- Performance monitoring integration
- Multi-dimensional data exploration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import time
import threading
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import psutil
import memory_profiler
from dataclasses import dataclass
import asyncio


@dataclass
class DashboardConfig:
    """Configuration class for dashboard settings"""
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    chart_height: int = 400
    enable_real_time: bool = True
    theme: str = "plotly_white"


class RealTimeDataManager:
    """Manages real-time data updates and streaming"""
    
    def __init__(self):
        self.data_streams = {}
        self.subscribers = {}
        self.is_running = False
        
    def add_data_stream(self, stream_id: str, data_source: callable):
        """Add a new data stream"""
        self.data_streams[stream_id] = {
            'source': data_source,
            'last_update': datetime.now(),
            'data': None
        }
        
    def subscribe(self, stream_id: str, callback: callable):
        """Subscribe to data stream updates"""
        if stream_id not in self.subscribers:
            self.subscribers[stream_id] = []
        self.subscribers[stream_id].append(callback)
        
    def start_streaming(self):
        """Start real-time data streaming"""
        self.is_running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
        
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_running = False
        
    def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running:
            for stream_id, stream_info in self.data_streams.items():
                try:
                    # Update data from source
                    new_data = stream_info['source']()
                    stream_info['data'] = new_data
                    stream_info['last_update'] = datetime.now()
                    
                    # Notify subscribers
                    if stream_id in self.subscribers:
                        for callback in self.subscribers[stream_id]:
                            callback(new_data)
                            
                except Exception as e:
                    st.error(f"Error updating stream {stream_id}: {str(e)}")
                    
            time.sleep(1)  # Update every second


class InteractiveChartBuilder:
    """Builds interactive charts with advanced features"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
    def create_time_series_chart(self, data: pd.DataFrame, x_col: str, y_col: str, 
                                title: str = "Time Series", **kwargs) -> go.Figure:
        """Create interactive time series chart"""
        fig = px.line(data, x=x_col, y=y_col, title=title, 
                     template=self.config.theme, **kwargs)
        
        fig.update_layout(
            height=self.config.chart_height,
            hovermode='x unified',
            showlegend=True
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
        
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """Create interactive correlation heatmap"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title=title,
                       template=self.config.theme)
        
        fig.update_layout(height=self.config.chart_height)
        return fig
        
    def create_distribution_chart(self, data: pd.DataFrame, column: str, 
                                 chart_type: str = "histogram") -> go.Figure:
        """Create distribution charts (histogram, box plot, violin plot)"""
        if chart_type == "histogram":
            fig = px.histogram(data, x=column, title=f"Distribution of {column}",
                             template=self.config.theme)
        elif chart_type == "box":
            fig = px.box(data, y=column, title=f"Box Plot of {column}",
                        template=self.config.theme)
        elif chart_type == "violin":
            fig = px.violin(data, y=column, title=f"Violin Plot of {column}",
                           template=self.config.theme)
        else:
            fig = px.histogram(data, x=column, title=f"Distribution of {column}",
                             template=self.config.theme)
            
        fig.update_layout(height=self.config.chart_height)
        return fig
        
    def create_scatter_matrix(self, data: pd.DataFrame, dimensions: List[str]) -> go.Figure:
        """Create interactive scatter plot matrix"""
        fig = px.scatter_matrix(data, dimensions=dimensions,
                               title="Scatter Plot Matrix",
                               template=self.config.theme)
        
        fig.update_layout(height=600)
        return fig
        
    def create_3d_scatter(self, data: pd.DataFrame, x: str, y: str, z: str, 
                         color: str = None) -> go.Figure:
        """Create 3D scatter plot"""
        fig = px.scatter_3d(data, x=x, y=y, z=z, color=color,
                           title=f"3D Scatter: {x} vs {y} vs {z}",
                           template=self.config.theme)
        
        fig.update_layout(height=self.config.chart_height)
        return fig


class PerformanceMonitor:
    """Monitors system performance and resource usage"""
    
    def __init__(self):
        self.metrics_history = []
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now()
        }
        
    def update_metrics_history(self):
        """Update metrics history"""
        current_metrics = self.get_current_metrics()
        self.metrics_history.append(current_metrics)
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
            
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to DataFrame"""
        if not self.metrics_history:
            return pd.DataFrame()
            
        return pd.DataFrame(self.metrics_history)
        
    def create_performance_charts(self) -> List[go.Figure]:
        """Create performance monitoring charts"""
        df = self.get_metrics_dataframe()
        if df.empty:
            return []
            
        charts = []
        
        # CPU Usage Chart
        cpu_fig = px.line(df, x='timestamp', y='cpu_percent',
                         title='CPU Usage Over Time',
                         labels={'cpu_percent': 'CPU Usage (%)'})
        cpu_fig.update_layout(height=300)
        charts.append(cpu_fig)
        
        # Memory Usage Chart
        memory_fig = px.line(df, x='timestamp', y='memory_percent',
                           title='Memory Usage Over Time',
                           labels={'memory_percent': 'Memory Usage (%)'})
        memory_fig.update_layout(height=300)
        charts.append(memory_fig)
        
        return charts


class DynamicDashboard:
    """Main Dynamic Dashboard class"""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.chart_builder = InteractiveChartBuilder(self.config)
        self.data_manager = RealTimeDataManager()
        self.performance_monitor = PerformanceMonitor()
        self.dashboard_state = {}
        
    def initialize(self):
        """Initialize dashboard components"""
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            
            # Start real-time data streaming if enabled
            if self.config.enable_real_time:
                self.data_manager.start_streaming()
                
            # Initialize performance monitoring
            self.performance_monitor.update_metrics_history()
            
    def create_dashboard_layout(self, data: pd.DataFrame) -> None:
        """Create the main dashboard layout"""
        st.title("🎯 Dynamic Analytics Dashboard")
        
        # Dashboard controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auto_refresh = st.checkbox("Auto Refresh", value=self.config.enable_real_time)
            
        with col2:
            refresh_interval = st.selectbox("Refresh Interval", 
                                          [1, 5, 10, 30, 60], 
                                          index=1)
            
        with col3:
            chart_theme = st.selectbox("Theme", 
                                     ["plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                                     index=0)
            
        with col4:
            if st.button("🔄 Refresh Now"):
                st.rerun()
                
        # Update configuration
        self.config.enable_real_time = auto_refresh
        self.config.refresh_interval = refresh_interval
        self.config.theme = chart_theme
        
        # Main dashboard content
        self._render_dashboard_content(data)
        
        # Auto-refresh mechanism
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
            
    def _render_dashboard_content(self, data: pd.DataFrame):
        """Render main dashboard content"""
        if data.empty:
            st.warning("No data available for visualization")
            return
            
        # Data overview section
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
            
        with col2:
            st.metric("Columns", len(data.columns))
            
        with col3:
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
            
        with col4:
            missing_values = data.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
            
        # Interactive visualizations section
        st.subheader("📈 Interactive Visualizations")
        
        # Chart type selection
        chart_tabs = st.tabs(["Time Series", "Distributions", "Correlations", "3D Analysis", "Performance"])
        
        with chart_tabs[0]:
            self._render_time_series_section(data)
            
        with chart_tabs[1]:
            self._render_distribution_section(data)
            
        with chart_tabs[2]:
            self._render_correlation_section(data)
            
        with chart_tabs[3]:
            self._render_3d_analysis_section(data)
            
        with chart_tabs[4]:
            self._render_performance_section()
            
    def _render_time_series_section(self, data: pd.DataFrame):
        """Render time series analysis section"""
        st.write("### Time Series Analysis")
        
        # Check for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not datetime_cols and not numeric_cols:
            st.info("No suitable columns found for time series analysis")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            if datetime_cols:
                x_col = st.selectbox("Time Column", datetime_cols)
            else:
                x_col = st.selectbox("X-axis", data.columns.tolist())
                
        with col2:
            y_col = st.selectbox("Value Column", numeric_cols)
            
        if x_col and y_col:
            fig = self.chart_builder.create_time_series_chart(data, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_distribution_section(self, data: pd.DataFrame):
        """Render distribution analysis section"""
        st.write("### Distribution Analysis")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns found for distribution analysis")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            selected_col = st.selectbox("Select Column", numeric_cols)
            
        with col2:
            chart_type = st.selectbox("Chart Type", ["histogram", "box", "violin"])
            
        if selected_col:
            fig = self.chart_builder.create_distribution_chart(data, selected_col, chart_type)
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_correlation_section(self, data: pd.DataFrame):
        """Render correlation analysis section"""
        st.write("### Correlation Analysis")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation analysis")
            return
            
        fig = self.chart_builder.create_correlation_heatmap(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix for selected columns
        if len(numeric_cols) >= 3:
            st.write("#### Scatter Plot Matrix")
            selected_dims = st.multiselect("Select Dimensions (max 5)", 
                                         numeric_cols, 
                                         default=numeric_cols[:3])
            
            if len(selected_dims) >= 2:
                fig_scatter = self.chart_builder.create_scatter_matrix(data, selected_dims[:5])
                st.plotly_chart(fig_scatter, use_container_width=True)
                
    def _render_3d_analysis_section(self, data: pd.DataFrame):
        """Render 3D analysis section"""
        st.write("### 3D Analysis")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            st.info("Need at least 3 numeric columns for 3D analysis")
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
            
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y")
            
        with col3:
            z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z")
            
        with col4:
            color_col = st.selectbox("Color by", [None] + data.columns.tolist(), key="3d_color")
            
        if x_col and y_col and z_col:
            fig = self.chart_builder.create_3d_scatter(data, x_col, y_col, z_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_performance_section(self):
        """Render performance monitoring section"""
        st.write("### System Performance")
        
        # Update performance metrics
        self.performance_monitor.update_metrics_history()
        
        # Current metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{current_metrics['cpu_percent']:.1f}%")
            
        with col2:
            st.metric("Memory Usage", f"{current_metrics['memory_percent']:.1f}%")
            
        with col3:
            st.metric("Memory Used", f"{current_metrics['memory_used_gb']:.2f} GB")
            
        # Performance charts
        charts = self.performance_monitor.create_performance_charts()
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
            
    def export_dashboard_config(self) -> Dict[str, Any]:
        """Export current dashboard configuration"""
        return {
            'config': {
                'refresh_interval': self.config.refresh_interval,
                'max_data_points': self.config.max_data_points,
                'chart_height': self.config.chart_height,
                'enable_real_time': self.config.enable_real_time,
                'theme': self.config.theme
            },
            'timestamp': datetime.now().isoformat()
        }
        
    def import_dashboard_config(self, config_dict: Dict[str, Any]):
        """Import dashboard configuration"""
        if 'config' in config_dict:
            config_data = config_dict['config']
            self.config.refresh_interval = config_data.get('refresh_interval', 5)
            self.config.max_data_points = config_data.get('max_data_points', 1000)
            self.config.chart_height = config_data.get('chart_height', 400)
            self.config.enable_real_time = config_data.get('enable_real_time', True)
            self.config.theme = config_data.get('theme', 'plotly_white')


# Global dashboard instance
dashboard_instance = None

def get_dashboard_instance() -> DynamicDashboard:
    """Get or create dashboard instance"""
    global dashboard_instance
    if dashboard_instance is None:
        dashboard_instance = DynamicDashboard()
        dashboard_instance.initialize()
    return dashboard_instance


def create_dashboard_tab(data: pd.DataFrame):
    """Create dashboard tab for Streamlit app"""
    dashboard = get_dashboard_instance()
    dashboard.create_dashboard_layout(data)