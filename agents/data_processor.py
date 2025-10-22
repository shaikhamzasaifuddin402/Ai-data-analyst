"""
Data preprocessing and cleaning agent
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Advanced data preprocessing and cleaning agent"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.processing_log = []
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality analysis"""
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'outliers': self._detect_outliers(df[col]).sum()
            }
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            quality_report['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
        
        return quality_report
    
    def clean_data(self, df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """Comprehensive data cleaning pipeline"""
        df_cleaned = df.copy()
        
        if config is None:
            config = {
                'remove_duplicates': True,
                'handle_missing': 'auto',
                'handle_outliers': 'cap',
                'normalize_text': True
            }
        
        logger.info(f"Starting data cleaning for dataset with shape {df.shape}")
        
        # Remove duplicates
        if config.get('remove_duplicates', True):
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            removed_duplicates = initial_rows - len(df_cleaned)
            if removed_duplicates > 0:
                self.processing_log.append(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, config.get('handle_missing', 'auto'))
        
        # Handle outliers
        if config.get('handle_outliers'):
            df_cleaned = self._handle_outliers(df_cleaned, method=config['handle_outliers'])
        
        # Normalize text columns
        if config.get('normalize_text', True):
            df_cleaned = self._normalize_text_columns(df_cleaned)
        
        logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        df_engineered = df.copy()
        
        # Date/time feature extraction
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            df_engineered[f'{col}_year'] = df_engineered[col].dt.year
            df_engineered[f'{col}_month'] = df_engineered[col].dt.month
            df_engineered[f'{col}_day'] = df_engineered[col].dt.day
            df_engineered[f'{col}_weekday'] = df_engineered[col].dt.weekday
            df_engineered[f'{col}_hour'] = df_engineered[col].dt.hour
        
        # Numeric feature interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Create polynomial features for first two numeric columns
            col1, col2 = numeric_cols[0], numeric_cols[1]
            df_engineered[f'{col1}_{col2}_interaction'] = df_engineered[col1] * df_engineered[col2]
            df_engineered[f'{col1}_squared'] = df_engineered[col1] ** 2
        
        # Binning continuous variables
        for col in numeric_cols:
            if df_engineered[col].nunique() > 10:  # Only bin if many unique values
                try:
                    df_engineered[f'{col}_binned'] = pd.cut(
                        df_engineered[col], 
                        bins=5, 
                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                    )
                except Exception:
                    # Skip binning if it fails (e.g., all values are the same)
                    pass
        
        self.processing_log.append(f"Feature engineering completed. Added {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # One-hot encode if few categories
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
            else:  # Label encode if many categories
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numeric features"""
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        self.scalers['main'] = scaler
        
        return df_scaled
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Handle missing values based on method"""
        df_imputed = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object']:
                    # Categorical imputation
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                else:
                    # Numeric imputation
                    if method == 'auto':
                        strategy = 'median' if abs(df[col].skew()) > 1 else 'mean'
                    else:
                        strategy = method
                    
                    imputer = SimpleImputer(strategy=strategy)
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                
                self.processing_log.append(f"Imputed {df[col].isnull().sum()} missing values in {col}")
        
        return df_imputed
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        df_outliers = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            outliers = self._detect_outliers(df[col])
            if outliers.sum() > 0:
                if method == 'cap':
                    # Cap outliers at 1st and 99th percentiles
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                    df_outliers[col] = df_outliers[col].clip(lower_bound, upper_bound)
                elif method == 'remove':
                    df_outliers = df_outliers[~outliers]
                
                self.processing_log.append(f"Handled {outliers.sum()} outliers in {col}")
        
        return df_outliers
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _normalize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text columns"""
        df_normalized = df.copy()
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            df_normalized[col] = df_normalized[col].astype(str).str.strip().str.title()
        
        return df_normalized
    
    def get_processing_summary(self) -> List[str]:
        """Get summary of all processing steps"""
        return self.processing_log.copy()

# Global instance
data_processor = DataProcessor()