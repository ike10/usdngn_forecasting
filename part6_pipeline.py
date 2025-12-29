"""
PART 6: COMPLETE PIPELINE - MAIN EXECUTION
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from part1_data_collection import DataCollector
from part2_preprocessing import DataPreprocessor, DataSplitter
from part3_information_analysis import run_information_analysis
from part4_models import ARIMAModel, LSTMModel, HybridARIMALSTM, RandomWalkModel
from part5_evaluation import ModelEvaluator, DieboldMarianoTest, RegimeEvaluator, SHAPExplainer

class USDNGNForecastingPipeline:
    def __init__(self, start_date='1995-01-01', end_date='2025-07-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.te_results = None
        self.feature_weights = None
        self.models = {}
        self.predictions = {}
        self.evaluation_results = {}
        self.feature_cols = ['brent_oil', 'mpr', 'cpi', 'oil_return', 'usdngn_volatility',
                            'usdngn_ma5', 'usdngn_ma20', 'rate_oil_ratio', 'mpr_change']
        
    def run(self, verbose=True):
        print("\n" + "=" * 70)
        print("USD-NGN EXCHANGE RATE FORECASTING PIPELINE")
        print("PhD Thesis: Oche Emmanuel Ike (242220011)")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Stage 1: Data Collection
        print("\n[STAGE 1] Data Collection")
        collector = DataCollector(self.start_date, self.end_date)
        self.raw_data = collector.collect_all_data()
        
        # Stage 2: Preprocessing
        print("\n[STAGE 2] Preprocessing")
        preprocessor = DataPreprocessor(self.raw_data)
        self.processed_data, _ = preprocessor.preprocess()
        splitter = DataSplitter()
        self.train_data, self.val_data, self.test_data = splitter.split(self.processed_data)
        
        # Stage 3: Information Analysis
        print("\n[STAGE 3] Information Analysis")
        info_results = run_information_analysis(self.train_data)
        self.te_results = info_results['te_results']
        self.feature_weights = info_results['feature_weights']
        
        # Stage 4: Model Training
        print("\n[STAGE 4] Model Training")
        available_features = [f for f in self.feature_cols if f in self.train_data.columns]
        X_train = self.train_data[available_features].values
        y_train = self.train_data['usdngn'].values
        X_val = self.val_data[available_features].values
        y_val = self.val_data['usdngn'].values
        X_test = self.test_data[available_features].values
        y_test = self.test_data['usdngn'].values
        
        te_weights = self.feature_weights['weight'].values if len(self.feature_weights) > 0 else None
        
        # Random Walk
        rw = RandomWalkModel().fit(y_train)
        self.models['Random Walk'] = rw
        rw_pred = np.roll(y_test, 1)
        rw_pred[0] = y_train[-1]
        self.predictions['Random Walk'] = rw_pred
        
        # ARIMA
        arima = ARIMAModel()
        arima.fit(y_train, verbose=verbose)
        self.models['ARIMA'] = arima
        self.predictions['ARIMA'] = arima.predict(len(y_test))
        
        # Hybrid
        hybrid = HybridARIMALSTM(arima_order=arima.best_order, feature_weights=te_weights)
        hybrid.fit(X_train, y_train, X_val, y_val, verbose=verbose)
        self.models['Hybrid'] = hybrid
        self.predictions['Hybrid'] = hybrid.predict(X_test)
        
        # Stage 5: Evaluation
        print("\n[STAGE 5] Evaluation")
        all_metrics = {}
        for name, pred in self.predictions.items():
            min_len = min(len(y_test), len(pred))
            metrics = ModelEvaluator.compute_all_metrics(y_test[:min_len], pred[:min_len])
            all_metrics[name] = metrics
            print(f"  {name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, DA={metrics['DA']:.1f}%")
        self.evaluation_results['metrics'] = all_metrics
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 70)
        print(f"Pipeline completed in {duration:.1f} seconds")
        print(f"Hybrid DA: {all_metrics['Hybrid']['DA']:.1f}% (vs Random Walk: {all_metrics['Random Walk']['DA']:.1f}%)")
        print("=" * 70)
        
        return self.get_results()
    
    def get_results(self):
        return {
            'data': {'raw': self.raw_data, 'processed': self.processed_data,
                    'train': self.train_data, 'val': self.val_data, 'test': self.test_data},
            'analysis': {'te_results': self.te_results, 'feature_weights': self.feature_weights},
            'models': self.models,
            'predictions': self.predictions,
            'evaluation': self.evaluation_results
        }

if __name__ == "__main__":
    pipeline = USDNGNForecastingPipeline()
    results = pipeline.run()
