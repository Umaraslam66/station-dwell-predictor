from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

class StationDwellPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.threshold_alerts = []

    def train(self, X_train, y_train):
        """Train multiple models and select the best one"""
        best_score = float('-inf')
        
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            avg_score = cv_scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                self.best_model = model
                self.best_model_name = name
        
        self.best_model.fit(X_train, y_train)
        return self.best_model_name, best_score

    def predict(self, X):
        """Make predictions using the best model"""
        return self.best_model.predict(X)

    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None:
            raise ValueError("Model hasn't been trained yet")
        return dict(zip(self.feature_names, self.best_model.feature_importances_))