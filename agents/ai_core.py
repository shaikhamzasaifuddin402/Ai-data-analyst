"""
Generative Analytics Core Module
Implements AI model integration for automated insights, predictions, and natural language processing
Based on the research paper specifications for the Generative AI-Powered Data Analytics Platform
"""

import pandas as pd
import numpy as np
import openai
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
import os
from datetime import datetime

# Optional heavy dependencies: guard imports to avoid noisy errors
try:
    import tensorflow as tf  # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    tf = None  # type: ignore
    TENSORFLOW_AVAILABLE = False
    logging.getLogger(__name__).warning("TensorFlow unavailable: %s", e)

try:
    from transformers import pipeline, AutoTokenizer, AutoModel  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    pipeline = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).warning("Transformers unavailable: %s", e)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerativeAnalyticsCore:
    """
    Core AI engine for generative analytics, natural language processing,
    and automated insights generation
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the Generative Analytics Core"""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Initialize NLP pipeline
        self.nlp_pipeline = None
        self.sentiment_analyzer = None
        self.text_generator = None
        
        # Initialize ML models
        self.ml_models = {}
        self.scalers = {}
        
        # Model performance tracking
        self.model_performance = {}
        
        # Initialize components
        self._initialize_nlp_components()
        
    def _initialize_nlp_components(self):
        """Initialize NLP components for text analysis"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers unavailable; skipping NLP component initialization")
                self.sentiment_analyzer = None
                self.text_generator = None
                return
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Initialize text generation (using a smaller model for efficiency)
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                max_length=100,
                num_return_sequences=1
            )
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")
            # Fallback to basic implementations
            self.sentiment_analyzer = None
            self.text_generator = None
    
    def process_natural_language_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process natural language queries and generate appropriate analytics
        
        Args:
            query: Natural language query from user
            data: DataFrame to analyze
            
        Returns:
            Dictionary containing analysis results and insights
        """
        try:
            # Analyze query intent
            intent = self._analyze_query_intent(query)
            
            # Generate appropriate analysis based on intent
            if intent['type'] == 'descriptive':
                result = self._generate_descriptive_analytics(data, intent)
            elif intent['type'] == 'predictive':
                result = self._generate_predictive_analytics(data, intent)
            elif intent['type'] == 'diagnostic':
                result = self._generate_diagnostic_analytics(data, intent)
            elif intent['type'] == 'prescriptive':
                result = self._generate_prescriptive_analytics(data, intent)
            else:
                result = self._generate_general_insights(data)
            
            # Add natural language explanation
            result['explanation'] = self._generate_explanation(result, query)
            result['query'] = query
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {str(e)}")
            return {
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent of a natural language query"""
        query_lower = query.lower()
        
        # Define intent keywords
        descriptive_keywords = ['describe', 'summary', 'overview', 'statistics', 'distribution']
        predictive_keywords = ['predict', 'forecast', 'future', 'will', 'estimate']
        diagnostic_keywords = ['why', 'cause', 'reason', 'correlation', 'relationship']
        prescriptive_keywords = ['recommend', 'suggest', 'optimize', 'improve', 'should']
        
        # Determine intent type
        if any(keyword in query_lower for keyword in predictive_keywords):
            intent_type = 'predictive'
        elif any(keyword in query_lower for keyword in diagnostic_keywords):
            intent_type = 'diagnostic'
        elif any(keyword in query_lower for keyword in prescriptive_keywords):
            intent_type = 'prescriptive'
        else:
            intent_type = 'descriptive'
        
        # Extract mentioned columns/features
        columns = []
        if hasattr(self, 'current_data') and self.current_data is not None:
            for col in self.current_data.columns:
                if col.lower() in query_lower:
                    columns.append(col)
        
        return {
            'type': intent_type,
            'columns': columns,
            'query': query
        }
    
    def _generate_descriptive_analytics(self, data: pd.DataFrame, intent: Dict) -> Dict[str, Any]:
        """Generate descriptive analytics"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            result = {
                'type': 'descriptive',
                'summary_statistics': {},
                'data_quality': {},
                'insights': []
            }
            
            # Basic statistics
            if len(numeric_cols) > 0:
                result['summary_statistics']['numeric'] = data[numeric_cols].describe().to_dict()
                
                # Identify outliers
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
                    if len(outliers) > 0:
                        result['insights'].append(f"Found {len(outliers)} outliers in {col}")
            
            # Categorical analysis
            if len(categorical_cols) > 0:
                result['summary_statistics']['categorical'] = {}
                for col in categorical_cols:
                    result['summary_statistics']['categorical'][col] = data[col].value_counts().to_dict()
            
            # Data quality assessment
            result['data_quality'] = {
                'total_rows': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'data_types': data.dtypes.astype(str).to_dict()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in descriptive analytics: {str(e)}")
            return {'error': str(e), 'type': 'descriptive'}
    
    def _generate_predictive_analytics(self, data: pd.DataFrame, intent: Dict) -> Dict[str, Any]:
        """Generate predictive analytics using ML models"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {
                    'error': 'Insufficient numeric columns for predictive modeling',
                    'type': 'predictive'
                }
            
            # Prepare data for modeling
            X = data[numeric_cols[:-1]]  # Features (all but last column)
            y = data[numeric_cols[-1]]   # Target (last numeric column)
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if len(np.unique(y)) < 10:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, predictions)
                
                result = {
                    'type': 'predictive',
                    'model_type': 'classification',
                    'accuracy': accuracy,
                    'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                    'predictions_sample': predictions[:10].tolist(),
                    'target_column': numeric_cols[-1]
                }
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, predictions)
                
                result = {
                    'type': 'predictive',
                    'model_type': 'regression',
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                    'predictions_sample': predictions[:10].tolist(),
                    'target_column': numeric_cols[-1]
                }
            
            # Store model for future use
            model_key = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.ml_models[model_key] = model
            self.scalers[model_key] = scaler
            
            result['model_id'] = model_key
            return result
            
        except Exception as e:
            logger.error(f"Error in predictive analytics: {str(e)}")
            return {'error': str(e), 'type': 'predictive'}
    
    def _generate_diagnostic_analytics(self, data: pd.DataFrame, intent: Dict) -> Dict[str, Any]:
        """Generate diagnostic analytics to understand relationships and causes"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            result = {
                'type': 'diagnostic',
                'correlations': {},
                'insights': []
            }
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = data[numeric_cols].corr()
                result['correlations'] = corr_matrix.to_dict()
                
                # Find strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            result['insights'].append(
                                f"Strong correlation ({corr_val:.3f}) between {col1} and {col2}"
                            )
            
            # Clustering analysis for pattern discovery
            if len(numeric_cols) >= 2:
                try:
                    X = data[numeric_cols].fillna(data[numeric_cols].mean())
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    result['clustering'] = {
                        'n_clusters': 3,
                        'cluster_centers': kmeans.cluster_centers_.tolist(),
                        'cluster_distribution': np.bincount(clusters).tolist()
                    }
                    
                    result['insights'].append(f"Identified 3 distinct patterns in the data")
                    
                except Exception as e:
                    logger.warning(f"Clustering analysis failed: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in diagnostic analytics: {str(e)}")
            return {'error': str(e), 'type': 'diagnostic'}
    
    def _generate_prescriptive_analytics(self, data: pd.DataFrame, intent: Dict) -> Dict[str, Any]:
        """Generate prescriptive analytics with recommendations"""
        try:
            result = {
                'type': 'prescriptive',
                'recommendations': [],
                'optimization_suggestions': []
            }
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Data quality recommendations
            missing_data = data.isnull().sum()
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = (missing_count / len(data)) * 100
                    if percentage > 10:
                        result['recommendations'].append(
                            f"Consider data imputation for {col} ({percentage:.1f}% missing)"
                        )
            
            # Feature engineering suggestions
            if len(numeric_cols) > 1:
                result['optimization_suggestions'].append(
                    "Consider feature scaling for machine learning models"
                )
                result['optimization_suggestions'].append(
                    "Explore feature interactions and polynomial features"
                )
            
            # Performance optimization suggestions
            if len(data) > 10000:
                result['optimization_suggestions'].append(
                    "Consider data sampling for faster processing of large datasets"
                )
            
            # Model selection recommendations
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
                unique_values = data[target_col].nunique()
                
                if unique_values < 10:
                    result['recommendations'].append(
                        f"Use classification models for {target_col} (categorical target)"
                    )
                else:
                    result['recommendations'].append(
                        f"Use regression models for {target_col} (continuous target)"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prescriptive analytics: {str(e)}")
            return {'error': str(e), 'type': 'prescriptive'}
    
    def _generate_general_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate general insights about the dataset"""
        try:
            result = {
                'type': 'general',
                'insights': [],
                'summary': {}
            }
            
            # Basic dataset information
            result['summary'] = {
                'rows': len(data),
                'columns': len(data.columns),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(data.select_dtypes(include=['object']).columns),
                'memory_usage': data.memory_usage(deep=True).sum()
            }
            
            # Generate insights
            if len(data) > 1000:
                result['insights'].append("Large dataset detected - consider sampling for faster analysis")
            
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if missing_percentage > 5:
                result['insights'].append(f"Dataset has {missing_percentage:.1f}% missing values")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating general insights: {str(e)}")
            return {'error': str(e), 'type': 'general'}
    
    def _generate_explanation(self, result: Dict[str, Any], query: str) -> str:
        """Generate natural language explanation of results"""
        try:
            if 'error' in result:
                return f"I encountered an error while processing your query: {result['error']}"
            
            result_type = result.get('type', 'unknown')
            
            if result_type == 'descriptive':
                return self._explain_descriptive_results(result)
            elif result_type == 'predictive':
                return self._explain_predictive_results(result)
            elif result_type == 'diagnostic':
                return self._explain_diagnostic_results(result)
            elif result_type == 'prescriptive':
                return self._explain_prescriptive_results(result)
            else:
                return "I've analyzed your data and generated insights based on your query."
                
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Analysis completed, but I couldn't generate a detailed explanation."
    
    def _explain_descriptive_results(self, result: Dict[str, Any]) -> str:
        """Explain descriptive analytics results"""
        explanations = []
        
        if 'summary_statistics' in result:
            explanations.append("I've calculated summary statistics for your dataset.")
        
        if 'insights' in result and result['insights']:
            explanations.append(f"Key findings: {'; '.join(result['insights'][:3])}")
        
        if 'data_quality' in result:
            total_rows = result['data_quality'].get('total_rows', 0)
            explanations.append(f"Your dataset contains {total_rows} rows.")
        
        return " ".join(explanations)
    
    def _explain_predictive_results(self, result: Dict[str, Any]) -> str:
        """Explain predictive analytics results"""
        model_type = result.get('model_type', 'unknown')
        
        if model_type == 'classification':
            accuracy = result.get('accuracy', 0)
            return f"I built a classification model with {accuracy:.1%} accuracy. The model can predict categories for your target variable."
        elif model_type == 'regression':
            rmse = result.get('rmse', 0)
            return f"I built a regression model with RMSE of {rmse:.3f}. The model can predict continuous values for your target variable."
        else:
            return "I've built a predictive model based on your data patterns."
    
    def _explain_diagnostic_results(self, result: Dict[str, Any]) -> str:
        """Explain diagnostic analytics results"""
        explanations = []
        
        if 'insights' in result and result['insights']:
            explanations.append("I found interesting relationships in your data:")
            explanations.extend(result['insights'][:2])
        
        if 'clustering' in result:
            n_clusters = result['clustering'].get('n_clusters', 0)
            explanations.append(f"I identified {n_clusters} distinct patterns in your dataset.")
        
        return " ".join(explanations)
    
    def _explain_prescriptive_results(self, result: Dict[str, Any]) -> str:
        """Explain prescriptive analytics results"""
        explanations = []
        
        recommendations = result.get('recommendations', [])
        if recommendations:
            explanations.append(f"I have {len(recommendations)} recommendations for improving your data and analysis.")
        
        optimizations = result.get('optimization_suggestions', [])
        if optimizations:
            explanations.append(f"I suggest {len(optimizations)} optimization strategies.")
        
        return " ".join(explanations) if explanations else "I've analyzed your data and prepared recommendations."
    
    def generate_synthetic_insights(self, data: pd.DataFrame, domain: str = "general") -> Dict[str, Any]:
        """
        Generate synthetic insights using AI models
        
        Args:
            data: DataFrame to analyze
            domain: Domain context (healthcare, finance, iot, general)
            
        Returns:
            Dictionary containing AI-generated insights
        """
        try:
            # Store current data for reference
            self.current_data = data
            
            # Generate domain-specific insights
            if domain.lower() == 'healthcare':
                return self._generate_healthcare_insights(data)
            elif domain.lower() == 'finance':
                return self._generate_finance_insights(data)
            elif domain.lower() == 'iot':
                return self._generate_iot_insights(data)
            else:
                return self._generate_general_domain_insights(data)
                
        except Exception as e:
            logger.error(f"Error generating synthetic insights: {str(e)}")
            return {
                'error': str(e),
                'domain': domain,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_healthcare_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate healthcare-specific insights"""
        insights = {
            'domain': 'healthcare',
            'insights': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        # Look for common healthcare patterns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if 'age' in col.lower():
                avg_age = data[col].mean()
                insights['insights'].append(f"Average age in dataset: {avg_age:.1f} years")
                
                if avg_age > 65:
                    insights['risk_factors'].append("High average age may indicate increased health risks")
            
            if any(term in col.lower() for term in ['pressure', 'bp', 'systolic', 'diastolic']):
                high_bp_count = len(data[data[col] > 140]) if col in data.columns else 0
                if high_bp_count > 0:
                    insights['risk_factors'].append(f"Found {high_bp_count} cases with elevated blood pressure")
        
        insights['recommendations'].extend([
            "Consider implementing preventive care programs",
            "Monitor high-risk patients more frequently",
            "Analyze treatment effectiveness patterns"
        ])
        
        return insights
    
    def _generate_finance_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate finance-specific insights"""
        insights = {
            'domain': 'finance',
            'insights': [],
            'risk_indicators': [],
            'recommendations': []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if 'amount' in col.lower() or 'value' in col.lower():
                high_value_threshold = data[col].quantile(0.95)
                high_value_count = len(data[data[col] > high_value_threshold])
                insights['insights'].append(f"Found {high_value_count} high-value transactions (>{high_value_threshold:.2f})")
                
                if high_value_count > len(data) * 0.1:
                    insights['risk_indicators'].append("Unusually high number of large transactions detected")
        
        # Check for potential fraud indicators
        if len(data) > 100:
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                insights['risk_indicators'].append(f"Found {duplicate_count} duplicate transactions")
        
        insights['recommendations'].extend([
            "Implement real-time fraud detection",
            "Set up automated alerts for unusual patterns",
            "Regular audit of high-value transactions"
        ])
        
        return insights
    
    def _generate_iot_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate IoT-specific insights"""
        insights = {
            'domain': 'iot',
            'insights': [],
            'anomalies': [],
            'recommendations': []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Analyze sensor data patterns
        for col in numeric_cols:
            if any(term in col.lower() for term in ['temp', 'temperature', 'humidity', 'pressure']):
                # Check for sensor anomalies
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
                
                if len(outliers) > 0:
                    insights['anomalies'].append(f"Detected {len(outliers)} anomalous readings in {col}")
        
        # Check data freshness (assuming timestamp column exists)
        timestamp_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            insights['insights'].append(f"Dataset spans {len(data)} sensor readings")
        
        insights['recommendations'].extend([
            "Implement predictive maintenance algorithms",
            "Set up real-time anomaly detection",
            "Monitor sensor health and calibration"
        ])
        
        return insights
    
    def _generate_general_domain_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate general domain insights"""
        return {
            'domain': 'general',
            'insights': [
                f"Dataset contains {len(data)} records with {len(data.columns)} features",
                f"Data quality score: {self._calculate_data_quality_score(data):.1%}"
            ],
            'recommendations': [
                "Explore data relationships with correlation analysis",
                "Consider feature engineering for better model performance",
                "Implement data validation and monitoring"
            ]
        }
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple data quality score"""
        try:
            # Factors: completeness, consistency, validity
            completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            consistency = 1 - (data.duplicated().sum() / len(data))
            
            # Simple average (can be made more sophisticated)
            quality_score = (completeness + consistency) / 2
            return max(0, min(1, quality_score))
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all trained models"""
        return self.model_performance.copy()
    
    def save_model(self, model_id: str, filepath: str) -> bool:
        """Save a trained model to disk"""
        try:
            if model_id in self.ml_models:
                joblib.dump({
                    'model': self.ml_models[model_id],
                    'scaler': self.scalers.get(model_id),
                    'performance': self.model_performance.get(model_id)
                }, filepath)
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str, model_id: str) -> bool:
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.ml_models[model_id] = model_data['model']
            if model_data.get('scaler'):
                self.scalers[model_id] = model_data['scaler']
            if model_data.get('performance'):
                self.model_performance[model_id] = model_data['performance']
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False