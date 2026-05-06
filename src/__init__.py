"""
USD-NGN Exchange Rate Forecasting Package
==========================================
Core modules for the forecasting system.
"""

from .data_collection import DataCollector
from .information_analysis import TransferEntropyAnalyzer, FeatureWeightComputer, run_information_analysis
from .preprocessing import DataPreprocessor, DataSplitter
from .models import ARIMAModel, RandomWalkModel, MovingAverageModel, HybridARIMALSTM
from .hybrid_model import MeanReversionModel, MeanReversionStreakModel, tune_mean_reversion_hyperparams
from .evaluation import ModelEvaluator, EnsembleUtils, WalkForwardValidator

__all__ = [
    'DataCollector',
    'TransferEntropyAnalyzer',
    'FeatureWeightComputer',
    'run_information_analysis',
    'DataPreprocessor',
    'DataSplitter',
    'ARIMAModel',
    'RandomWalkModel',
    'MovingAverageModel',
    'HybridARIMALSTM',
    'MeanReversionModel',
    'MeanReversionStreakModel',
    'ModelEvaluator',
    'EnsembleUtils',
    'WalkForwardValidator',
    'tune_mean_reversion_hyperparams',
]
