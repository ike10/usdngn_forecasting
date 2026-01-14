"""
PART 2: DATA PREPROCESSING MODULE
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class DataPreprocessor:
    def __init__(self, df):
        self.raw_data = df.copy()
        self.processed_data = None
        self.scalers = {}
        self.stationarity_results = {}
        self.feature_names = []
        
    def engineer_features(self, df):
        print("\n[Preprocessing] Engineering features...")
        # Eq 3.1: Log Returns
        df['usdngn_return'] = np.log(df['usdngn'] / df['usdngn'].shift(1))
        df['oil_return'] = np.log(df['brent_oil'] / df['brent_oil'].shift(1))
        # Eq 3.2: Moving Averages
        for var in ['usdngn', 'brent_oil']:
            df[f'{var}_ma5'] = df[var].rolling(window=5, min_periods=1).mean()
            df[f'{var}_ma20'] = df[var].rolling(window=20, min_periods=1).mean()
            df[f'{var}_ma60'] = df[var].rolling(window=60, min_periods=1).mean()
        # Eq 3.3: Volatility
        df['usdngn_volatility'] = df['usdngn_return'].rolling(window=20, min_periods=5).std()
        df['oil_volatility'] = df['oil_return'].rolling(window=20, min_periods=5).std()
        # Lag Features
        for lag in [1, 5, 10]:
            df[f'usdngn_lag{lag}'] = df['usdngn'].shift(lag)
            df[f'oil_lag{lag}'] = df['brent_oil'].shift(lag)
        # Derived Features
        df['rate_oil_ratio'] = df['usdngn'] / df['brent_oil']
        df['mpr_change'] = df['mpr'].diff()
        df['cpi_momentum'] = df['cpi'] - df['cpi'].shift(20)
        df['usdngn_trend'] = (df['usdngn'] - df['usdngn_ma20']) / df['usdngn_ma20']
        df['oil_trend'] = (df['brent_oil'] - df['brent_oil_ma20']) / df['brent_oil_ma20']
        df['usdngn_roc5'] = (df['usdngn'] - df['usdngn'].shift(5)) / df['usdngn'].shift(5)
        df['oil_roc5'] = (df['brent_oil'] - df['brent_oil'].shift(5)) / df['brent_oil'].shift(5)
        return df
    
    def test_stationarity(self, series, name):
        if not STATSMODELS_AVAILABLE:
            return {'name': name, 'is_stationary': None}
        series_clean = series.dropna()
        if len(series_clean) < 20:
            return {'name': name, 'is_stationary': None}
        try:
            adf_result = adfuller(series_clean, autolag='BIC')
            adf_p = adf_result[1]
        except:
            adf_p = 1.0
        try:
            kpss_result = kpss(series_clean, regression='c', nlags='auto')
            kpss_p = kpss_result[1]
        except:
            kpss_p = 0.05
        is_stationary = adf_p < 0.05 and kpss_p > 0.05
        return {'name': name, 'adf_p': adf_p, 'kpss_p': kpss_p, 'is_stationary': is_stationary}
    
    def preprocess(self, normalize=False):
        print("\n" + "=" * 70)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 70)
        df = self.raw_data.copy()
        df = df.ffill().bfill()
        df = self.engineer_features(df)
        df = df.dropna()
        print(f"\nFinal dataset shape: {df.shape}")
        test_vars = ['usdngn', 'brent_oil', 'usdngn_return', 'oil_return']
        stationarity_results = []
        for var in test_vars:
            if var in df.columns:
                result = self.test_stationarity(df[var], var)
                stationarity_results.append(result)
                self.stationarity_results[var] = result
        self.feature_names = [col for col in df.columns if col != 'usdngn']
        self.processed_data = df
        return df, pd.DataFrame(stationarity_results)

class DataSplitter:
    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, df):
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        print(f"\nTraining: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")
        return train, val, test

if __name__ == "__main__":
    from part1_data_collection import DataCollector
    collector = DataCollector()
    raw_df = collector.collect_all_data()
    preprocessor = DataPreprocessor(raw_df)
    processed_df, stationarity = preprocessor.preprocess()
    splitter = DataSplitter()
    train, val, test = splitter.split(processed_df)
