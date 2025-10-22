"""
Synthetic data generation utilities
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

fake = Faker()

class DataGenerator:
    """Generate synthetic datasets for different domains"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        Faker.seed(seed)
        random.seed(seed)
    
    def generate_healthcare_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic healthcare patient data"""
        data = []
        
        for _ in range(n_samples):
            age = np.random.randint(18, 90)
            gender = np.random.choice(['Male', 'Female'])
            
            # Generate correlated health metrics
            bmi_base = np.random.normal(25, 5)
            bmi = max(15, min(45, bmi_base))
            
            # Blood pressure correlated with age and BMI
            systolic = 90 + age * 0.5 + (bmi - 25) * 2 + np.random.normal(0, 10)
            diastolic = 60 + age * 0.3 + (bmi - 25) * 1.5 + np.random.normal(0, 8)
            
            # Cholesterol correlated with age and BMI
            cholesterol = 150 + age * 1.2 + (bmi - 25) * 3 + np.random.normal(0, 20)
            
            # Risk-based diagnosis
            risk_score = (age - 30) * 0.1 + (bmi - 25) * 0.2 + (systolic - 120) * 0.05
            diagnosis_prob = 1 / (1 + np.exp(-risk_score * 0.1))
            diagnosis = 'High Risk' if np.random.random() < diagnosis_prob else 'Low Risk'
            
            data.append({
                'patient_id': f'P{_:06d}',
                'age': int(age),
                'gender': gender,
                'bmi': round(bmi, 2),
                'systolic_bp': round(max(80, systolic), 1),
                'diastolic_bp': round(max(50, diastolic), 1),
                'cholesterol': round(max(100, cholesterol), 1),
                'diagnosis': diagnosis,
                'last_visit': fake.date_between(start_date='-2y', end_date='today')
            })
        
        return pd.DataFrame(data)
    
    def generate_finance_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic financial transaction data"""
        data = []
        
        for _ in range(n_samples):
            customer_id = f'C{_:06d}'
            credit_score = np.random.randint(300, 850)
            
            # Account balance correlated with credit score
            balance_base = (credit_score - 300) * 100 + np.random.normal(0, 5000)
            account_balance = max(0, balance_base)
            
            # Transaction amount based on balance
            max_transaction = min(account_balance * 0.3, 10000)
            transaction_amount = np.random.exponential(max_transaction * 0.1)
            
            transaction_type = np.random.choice([
                'Purchase', 'Transfer', 'Withdrawal', 'Deposit', 'Payment'
            ], p=[0.4, 0.2, 0.15, 0.15, 0.1])
            
            data.append({
                'customer_id': customer_id,
                'transaction_id': f'T{_:08d}',
                'transaction_amount': round(transaction_amount, 2),
                'transaction_type': transaction_type,
                'account_balance': round(account_balance, 2),
                'credit_score': credit_score,
                'transaction_date': fake.date_time_between(start_date='-1y', end_date='now'),
                'merchant_category': fake.company_suffix()
            })
        
        return pd.DataFrame(data)
    
    def generate_iot_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic IoT sensor data"""
        data = []
        devices = [f'IOT_{i:03d}' for i in range(1, 21)]  # 20 devices
        
        base_time = datetime.now() - timedelta(days=30)
        
        for _ in range(n_samples):
            device_id = np.random.choice(devices)
            timestamp = base_time + timedelta(
                seconds=np.random.randint(0, 30 * 24 * 3600)
            )
            
            # Simulate daily temperature cycle
            hour = timestamp.hour
            temp_base = 20 + 10 * np.sin((hour - 6) * np.pi / 12)
            temperature = temp_base + np.random.normal(0, 2)
            
            # Humidity inversely correlated with temperature
            humidity = 70 - (temperature - 20) * 2 + np.random.normal(0, 5)
            humidity = max(20, min(95, humidity))
            
            # Pressure with some correlation to temperature
            pressure = 1013 + (temperature - 20) * 0.5 + np.random.normal(0, 3)
            
            # Device status based on sensor readings
            if temperature > 35 or humidity > 90 or pressure < 1000:
                status = 'Warning'
            elif temperature < 5 or humidity < 30 or pressure > 1030:
                status = 'Alert'
            else:
                status = 'Normal'
            
            data.append({
                'device_id': device_id,
                'timestamp': timestamp,
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'pressure': round(pressure, 2),
                'status': status,
                'battery_level': np.random.randint(10, 100),
                'signal_strength': np.random.randint(-80, -30)
            })
        
        return pd.DataFrame(data)

    def generate_sales_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic retail sales data"""
        data = []
        regions = ['North', 'South', 'East', 'West']
        products = ['Laptop', 'Phone', 'Tablet', 'Accessories']
        base_prices = {
            'Laptop': 1200,
            'Phone': 800,
            'Tablet': 700,
            'Accessories': 150
        }
        for i in range(n_samples):
            product = np.random.choice(products)
            region = np.random.choice(regions)
            quantity = np.random.randint(1, 5)
            price = base_prices[product] + np.random.normal(0, base_prices[product] * 0.1)
            sales_amount = max(20.0, price * quantity + np.random.normal(0, 50))
            customer_age = np.random.randint(18, 70)
            marketing_spend = round(max(0, np.random.normal(150, 50)), 2)
            satisfaction = round(np.clip(np.random.normal(4.0, 0.5), 1.0, 5.0), 1)
            date = fake.date_between(start_date='-180d', end_date='today')
            data.append({
                'Date': date,
                'Region': region,
                'Product': product,
                'Sales_Amount': round(sales_amount, 2),
                'Quantity': int(quantity),
                'Customer_Age': int(customer_age),
                'Marketing_Spend': float(marketing_spend),
                'Customer_Satisfaction': float(satisfaction)
            })
        return pd.DataFrame(data)

    def generate_customer_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic customer profile data"""
        data = []
        genders = ['Male', 'Female']
        segments = ['Standard', 'Premium', 'Enterprise']
        for i in range(n_samples):
            age = np.random.randint(18, 75)
            gender = np.random.choice(genders)
            signup_date = fake.date_between(start_date='-2y', end_date='today')
            churn = np.random.choice([0, 1], p=[0.8, 0.2])
            base_ltv = age * 100 + np.random.normal(0, 1000)
            lifetime_value = max(100, base_ltv * (1.2 if churn == 0 else 0.6))
            data.append({
                'customer_id': f'C{i:06d}',
                'name': fake.name(),
                'age': int(age),
                'gender': gender,
                'city': fake.city(),
                'segment': np.random.choice(segments),
                'signup_date': signup_date,
                'churn': int(churn),
                'lifetime_value': round(float(lifetime_value), 2)
            })
        return pd.DataFrame(data)

    def generate_sample_data(self, data_type: str, n_samples: int = 1000) -> pd.DataFrame:
        """Convenience wrapper to generate data by type string"""
        dt = (data_type or '').strip().lower()
        if dt in ('sales_data', 'sales', 'sales-data'):
            return self.generate_sales_data(n_samples)
        if dt in ('customer_data', 'customers', 'customer'):
            return self.generate_customer_data(n_samples)
        if dt in ('financial_data', 'financial', 'finance'):
            return self.generate_finance_data(n_samples)
        if dt in ('iot_sensor_data', 'iot', 'sensor', 'iot-data'):
            return self.generate_iot_data(n_samples)
        if dt in ('healthcare_data', 'healthcare', 'medical'):
            return self.generate_healthcare_data(n_samples)
        raise ValueError(f"Unsupported data type: {data_type}")

    def generate_like(self, df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic data resembling the uploaded dataset's schema and distributions.
        - Numeric columns: sample from normal with source mean/std; preserve int/float.
        - Categorical/object: sample based on observed value frequencies.
        - Datetime: sample uniformly within observed min/max.
        - Boolean: sample by observed True/False ratio.
        - If a column has no variance or too few values, fallback to sampling original values.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=[])
        result = {}
        for col in df.columns:
            s = df[col].dropna()
            # Fallback for empty column
            if s.empty:
                result[col] = [None] * n_samples
                continue
            dtype = s.dtype
            try:
                # Datetime
                if np.issubdtype(dtype, np.datetime64):
                    # convert to datetime if not
                    s_dt = pd.to_datetime(s, errors='coerce').dropna()
                    if s_dt.empty:
                        # sample original values
                        values = np.random.choice(s.astype(str).values, size=n_samples, replace=True)
                        result[col] = values
                    else:
                        start, end = s_dt.min(), s_dt.max()
                        delta = (end - start).total_seconds()
                        if delta <= 0:
                            values = [start] * n_samples
                        else:
                            rand_secs = np.random.uniform(0, delta, size=n_samples)
                            values = [start + pd.Timedelta(seconds=float(x)) for x in rand_secs]
                        result[col] = values
                # Boolean
                elif s.dtype == bool or pd.api.types.is_bool_dtype(s):
                    p_true = s.mean() if s.dtype == bool else (s.astype(int).mean())
                    values = np.random.choice([True, False], size=n_samples, p=[p_true, 1 - p_true])
                    result[col] = values
                # Numeric
                elif pd.api.types.is_numeric_dtype(dtype):
                    mean = float(s.mean())
                    std = float(s.std()) if s.std() > 0 else max(1e-6, abs(mean) * 0.1)
                    gen = np.random.normal(mean, std, size=n_samples)
                    if pd.api.types.is_integer_dtype(dtype):
                        gen = np.round(gen).astype(int)
                    else:
                        gen = gen.astype(float)
                    result[col] = gen
                # Categorical / object
                else:
                    # Use frequency-based sampling of existing values
                    vc = s.value_counts(normalize=True)
                    categories = vc.index.tolist()
                    probs = vc.values
                    # In some rare cases, value_counts may not return probs summing to 1; normalize
                    if probs.sum() <= 0:
                        categories = s.astype(str).unique().tolist()
                        probs = np.ones(len(categories)) / len(categories)
                    values = np.random.choice(categories, size=n_samples, replace=True, p=probs)
                    result[col] = values
            except Exception:
                # Fallback: sample original values
                values = np.random.choice(s.values, size=n_samples, replace=True)
                result[col] = values
        return pd.DataFrame(result)

# Global instance
data_generator = DataGenerator()