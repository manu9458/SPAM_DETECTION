"""
Spam Detection Package
======================

This package contains the source code for the Spam Detection System.
It includes modules for data loading, preprocessing, model training, and evaluation.
"""

from .data_loader import DataLoader
from .preprocessor import TextPreprocessor
from .model_trainer import SpamModelTrainer
from .evaluator import ModelEvaluator

__version__ = '1.0.0'
__author__ = 'Manu'
