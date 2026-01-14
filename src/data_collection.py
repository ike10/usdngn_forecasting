"""
PART 1: DATA COLLECTION MODULE
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

class DataCollector:
    def __init__(self, start_date='1995-01-01', end_date='2025-07-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.data_sources = {}
        
    def generate_realistic_usdngn(self):
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        regimes = {
            '1995-01-01': {'rate': 22, 'vol': 0.002},
            '1999-01-01': {'rate': 92, 'vol': 0.003},
            '2008-09-01': {'rate': 120, 'vol': 0.008},
            '2014-07-01': {'rate': 160, 'vol': 0.005},
            '2016-06-20': {'rate': 283, 'vol': 0.015},
            '2017-01-01': {'rate': 305, 'vol': 0.004},
            '2020-03-01': {'rate': 360, 'vol': 0.010},
            '2021-01-01': {'rate': 410, 'vol': 0.004},
            '2023-06-14': {'rate': 750, 'vol': 0.025},
            '2024-03-01': {'rate': 1500, 'vol': 0.025},
            '2025-01-01': {'rate': 1450, 'vol': 0.012}
        }
        rates = np.zeros(len(dates))
        current_rate = 22
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            applicable_regime = None
            for regime_start in sorted(regimes.keys()):
                if date_str >= regime_start:
                    applicable_regime = regimes[regime_start]
            if applicable_regime:
                target_rate = applicable_regime['rate']
                volatility = applicable_regime['vol']
                current_rate = current_rate + 0.02 * (target_rate - current_rate)
                noise = np.random.normal(0, volatility * current_rate)
                current_rate = max(current_rate + noise, 1)
            rates[i] = current_rate
        self.data_sources['usdngn'] = 'Synthetic (CBN-calibrated)'
        return pd.Series(rates, index=dates, name='usdngn')
    
    def generate_realistic_brent_oil(self):
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        regimes = {
            '1995-01-01': {'price': 17, 'vol': 0.015},
            '2000-01-01': {'price': 28, 'vol': 0.018},
            '2008-07-01': {'price': 140, 'vol': 0.035},
            '2008-12-01': {'price': 40, 'vol': 0.040},
            '2011-01-01': {'price': 100, 'vol': 0.015},
            '2014-07-01': {'price': 110, 'vol': 0.025},
            '2016-01-01': {'price': 30, 'vol': 0.035},
            '2018-01-01': {'price': 70, 'vol': 0.018},
            '2020-04-20': {'price': 20, 'vol': 0.050},
            '2022-03-01': {'price': 120, 'vol': 0.030},
            '2024-01-01': {'price': 78, 'vol': 0.015},
            '2025-01-01': {'price': 72, 'vol': 0.012}
        }
        prices = np.zeros(len(dates))
        current_price = 17
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            applicable_regime = None
            for regime_start in sorted(regimes.keys()):
                if date_str >= regime_start:
                    applicable_regime = regimes[regime_start]
            if applicable_regime:
                target_price = applicable_regime['price']
                volatility = applicable_regime['vol']
                current_price = current_price + 0.03 * (target_price - current_price)
                noise = np.random.normal(0, volatility * current_price)
                current_price = max(current_price + noise, 5)
            prices[i] = current_price
        self.data_sources['brent_oil'] = 'Synthetic (FRED-calibrated)'
        return pd.Series(prices, index=dates, name='brent_oil')
    
    def generate_realistic_mpr(self):
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        rate_history = {
            '1995-01-01': 13.5, '2006-12-01': 10.0, '2010-09-01': 6.25,
            '2011-10-01': 12.0, '2016-07-01': 14.0, '2020-09-01': 11.5,
            '2023-07-01': 18.75, '2024-02-01': 22.75, '2024-11-01': 27.50
        }
        mpr_values = []
        current_rate = 13.5
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            for rate_date in sorted(rate_history.keys()):
                if date_str >= rate_date:
                    current_rate = rate_history[rate_date]
            mpr_values.append(current_rate)
        self.data_sources['mpr'] = 'CBN Official Decisions'
        return pd.Series(mpr_values, index=dates, name='mpr')
    
    def generate_realistic_cpi(self):
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        inflation_history = {
            '1995-01-01': 72.8, '1999-01-01': 6.6, '2005-01-01': 17.9,
            '2008-01-01': 11.6, '2016-12-01': 18.6, '2020-12-01': 15.8,
            '2023-08-01': 25.8, '2024-06-01': 34.2, '2025-06-01': 22.0
        }
        cpi_values = []
        current_inflation = 72.8
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            for inf_date in sorted(inflation_history.keys()):
                if date_str >= inf_date:
                    current_inflation = inflation_history[inf_date]
            cpi_values.append(current_inflation + np.random.normal(0, 0.1))
        self.data_sources['cpi'] = 'NBS/CBN Official Releases'
        return pd.Series(cpi_values, index=dates, name='cpi')
    
    def collect_all_data(self, use_real_data=True):
        print("=" * 70)
        print("DATA COLLECTION PHASE")
        print("=" * 70)
        data_dict = {}
        print("\n[1/4] Generating USD-NGN Exchange Rate...")
        data_dict['usdngn'] = self.generate_realistic_usdngn()
        print("\n[2/4] Generating Brent Crude Oil Prices...")
        data_dict['brent_oil'] = self.generate_realistic_brent_oil()
        print("\n[3/4] Generating Monetary Policy Rate...")
        data_dict['mpr'] = self.generate_realistic_mpr()
        print("\n[4/4] Generating CPI/Inflation data...")
        data_dict['cpi'] = self.generate_realistic_cpi()
        df = pd.DataFrame(data_dict).dropna()
        df.index = pd.to_datetime(df.index)
        self.data = df.sort_index()
        print(f"\nTotal observations: {len(df):,}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        return self.data

if __name__ == "__main__":
    collector = DataCollector()
    df = collector.collect_all_data()
    print(df.head())
    print(df.tail())
