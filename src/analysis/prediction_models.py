"""
Machine learning models for prediction
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

class PredictionModel:
    def __init__(self, model_type='linear'):
        """
        Initialize prediction model
        
        Args:
            model_type (str): Type of model to use ('linear', 'rf' for Random Forest, etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Prepare data for modeling
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            tuple: Scaled training and testing data
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        if self.model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        elif self.model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
