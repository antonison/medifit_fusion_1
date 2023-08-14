# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:54:08 2023

@author: anton
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import convolve
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import pandas as pd

# Custom transformer for baseline correction
class BaselineCorrection(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.sub(X.mean(axis=1), axis=0)

# Custom transformer for Standard Normal Variate (SNV)
class StandardNormalVariate(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

# Custom transformer for Savitzky-Golay filter
class SavitzkyGolayFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda row: pd.Series(savgol_filter(row, 15, 1)), axis=1)

# Singular Value Decomposition (SVD)
class SingularValueDecomposition(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        return U @ np.diag(S) @ VT

# Second Derivative
class SecondDerivative(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        kernel = np.array([1, -2, 1])
        return X.apply(lambda row: pd.Series(convolve(row, kernel, mode='nearest')), axis=1)

# Negative Logarithm
class NegativeLogarithm(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return -np.log(X)

# Z-Score Normalization
class ZScoreNormalization(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

# Min-Max Normalization
class MinMaxNormalization(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

# Mean Centering
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X - X.mean(axis=0)