import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings('ignore')

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
        
    def generate_dummy_data(self, n_samples=1000):
        """Generate synthetic station data with more realistic patterns"""
        np.random.seed(42)
        
        # Time-based features
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        hours = dates.hour
        
        # Generate features with more realistic patterns
        passenger_volume = np.zeros(n_samples)
        for i, hour in enumerate(hours):
            # Morning peak (7-9 AM)
            if 7 <= hour <= 9:
                passenger_volume[i] = np.random.normal(800, 150)
            # Evening peak (16-18 PM)
            elif 16 <= hour <= 18:
                passenger_volume[i] = np.random.normal(900, 200)
            # Normal hours
            else:
                passenger_volume[i] = np.random.normal(400, 100)

        platform_length = np.random.choice([100, 150, 200, 250], n_samples)
        peak_hour = np.where((hours >= 7) & (hours <= 9) | 
                           (hours >= 16) & (hours <= 18), 1, 0)
        
        # Weather with seasonal patterns
        month = dates.month
        weather_condition = np.zeros(n_samples)
        
        # Winter months (December, January, February)
        winter_mask = month.isin([12, 1, 2])
        weather_condition[winter_mask] = np.random.choice(
            [0, 1, 2], 
            size=winter_mask.sum(), 
            p=[0.4, 0.3, 0.3]
        )
        
        # Spring/Fall months
        spring_fall_mask = month.isin([3, 4, 5, 9, 10, 11])
        weather_condition[spring_fall_mask] = np.random.choice(
            [0, 1, 2], 
            size=spring_fall_mask.sum(), 
            p=[0.6, 0.3, 0.1]
        )
        
        # Summer months
        summer_mask = month.isin([6, 7, 8])
        weather_condition[summer_mask] = np.random.choice(
            [0, 1, 2], 
            size=summer_mask.sum(), 
            p=[0.8, 0.2, 0.0]
        )
        
        # Special events (randomly occurring)
        special_event = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        
        # Station complexity (platform layout, entrance/exit count)
        station_complexity = np.random.choice([1, 2, 3], n_samples)  # 1=simple, 2=medium, 3=complex
        
        # Generate target (dwell time) with realistic relationships
        base_dwell_time = 60  # Base dwell time in seconds
        dwell_time = (
            base_dwell_time +
            passenger_volume * 0.1 +  # More passengers = longer dwell time
            peak_hour * 30 +  # Peak hours add 30 seconds
            weather_condition * 15 +  # Bad weather adds delay
            special_event * 45 +  # Special events add 45 seconds
            station_complexity * 10 +  # Complex stations add delay
            np.random.normal(0, 10, n_samples)  # Random noise
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'datetime': dates,
            'passenger_volume': passenger_volume,
            'platform_length': platform_length,
            'peak_hour': peak_hour,
            'weather_condition': weather_condition,
            'special_event': special_event,
            'station_complexity': station_complexity,
            'hour': hours,
            'month': month,
            'day_of_week': dates.dayofweek,
            'dwell_time': dwell_time
        })
        
        self.feature_names = data.columns.drop(['datetime', 'dwell_time']).tolist()
        return data

    def detect_anomalies(self, predictions, actual, threshold=2):
        """Detect anomalous dwell times"""
        residuals = actual - predictions
        std_residuals = np.std(residuals)
        anomalies = np.abs(residuals) > (threshold * std_residuals)
        return anomalies

    def generate_threshold_alerts(self, data, predictions):
        """Generate alerts based on various thresholds"""
        self.threshold_alerts = []
        
        # High passenger volume alert
        high_volume_mask = data['passenger_volume'] > data['passenger_volume'].quantile(0.95)
        if high_volume_mask.any():
            self.threshold_alerts.append({
                'type': 'High Passenger Volume',
                'count': high_volume_mask.sum(),
                'avg_dwell_impact': predictions[high_volume_mask].mean() - predictions.mean()
            })
        
        # Weather impact alert
        bad_weather_mask = data['weather_condition'] > 0
        if bad_weather_mask.any():
            self.threshold_alerts.append({
                'type': 'Weather Impact',
                'count': bad_weather_mask.sum(),
                'avg_dwell_impact': predictions[bad_weather_mask].mean() - predictions.mean()
            })
        
        # Special event impact
        special_event_mask = data['special_event'] == 1
        if special_event_mask.any():
            self.threshold_alerts.append({
                'type': 'Special Event',
                'count': special_event_mask.sum(),
                'avg_dwell_impact': predictions[special_event_mask].mean() - predictions.mean()
            })

    def prepare_data(self, data):
        """Prepare data for training"""
        X = data[self.feature_names]
        y = data['dwell_time']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        """Train multiple models and select the best one"""
        best_score = float('-inf')
        
        for name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            avg_score = cv_scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                self.best_model = model
                self.best_model_name = name
        
        # Train the best model on the full training set
        self.best_model.fit(X_train, y_train)
        return self.best_model_name, best_score

    def predict(self, X):
        """Make predictions using the best model"""
        return self.best_model.predict(X)

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        predictions = self.predict(X_test)
        
        return {
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions),
            'Feature Importance': dict(zip(
                self.feature_names,
                self.best_model.feature_importances_
            )),
            'Anomalies': self.detect_anomalies(predictions, y_test).sum()
        }

    def create_visualization_dashboard(self, data, true_values, predictions):
        """Create a comprehensive visualization dashboard"""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Actual vs Predicted Dwell Times',
                'Feature Importance',
                'Dwell Time Distribution',
                'Passenger Volume vs Dwell Time',
                'Hourly Dwell Time Pattern',
                'Weather Impact on Dwell Time'
            )
        )
        
        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=true_values, y=predictions, mode='markers',
                      name='Predictions'),
            row=1, col=1
        )
        
        # 2. Feature Importance
        importance = self.best_model.feature_importances_
        fig.add_trace(
            go.Bar(x=self.feature_names, y=importance,
                  name='Feature Importance'),
            row=1, col=2
        )
        
        # 3. Dwell Time Distribution
        fig.add_trace(
            go.Histogram(x=predictions, name='Predicted Dwell Times',
                        nbinsx=30),
            row=2, col=1
        )
        
        # 4. Passenger Volume vs Dwell Time
        fig.add_trace(
            go.Scatter(x=data['passenger_volume'], y=predictions,
                      mode='markers', name='Volume Impact'),
            row=2, col=2
        )
        
        # 5. Hourly Pattern
        hourly_avg = pd.DataFrame({
            'hour': data['hour'],
            'dwell_time': predictions
        }).groupby('hour').mean()
        
        fig.add_trace(
            go.Scatter(x=hourly_avg.index, y=hourly_avg.dwell_time,
                      name='Hourly Pattern', mode='lines+markers'),
            row=3, col=1
        )
        
        # 6. Weather Impact
        weather_impact = pd.DataFrame({
            'weather': data['weather_condition'],
            'dwell_time': predictions
        }).groupby('weather').mean()
        
        fig.add_trace(
            go.Bar(x=['Good', 'Rain', 'Snow'], 
                  y=weather_impact.dwell_time,
                  name='Weather Impact'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(height=1000, width=1200, showlegend=False,
                         title_text="Station Dwell Time Analysis Dashboard")
        
        return fig

def main():
    # Initialize predictor
    predictor = StationDwellPredictor()
    
    # Generate and prepare data
    print("Generating synthetic data...")
    data = predictor.generate_dummy_data(n_samples=5000)
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    
    # Train model
    print("\nTraining models...")
    best_model, cv_score = predictor.train(X_train, y_train)
    print(f"Best model: {best_model} (CV Score: {cv_score:.3f})")
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Generate alerts
    predictor.generate_threshold_alerts(
        data.iloc[len(data)-len(predictions):].reset_index(drop=True), 
        predictions
    )
    
    # Evaluate model
    evaluation = predictor.evaluate_model(X_test, y_test)
    print("\nModel Evaluation:")
    print(f"RMSE: {evaluation['RMSE']:.2f} seconds")
    print(f"MAE: {evaluation['MAE']:.2f} seconds")
    print(f"R2 Score: {evaluation['R2']:.2f}")
    print(f"Anomalies detected: {evaluation['Anomalies']}")
    
    print("\nFeature Importance:")
    for feature, importance in evaluation['Feature Importance'].items():
        print(f"{feature}: {importance:.3f}")
    
    print("\nThreshold Alerts:")
    for alert in predictor.threshold_alerts:
        print(f"{alert['type']}:")
        print(f"  Occurrences: {alert['count']}")
        print(f"  Average Impact: {alert['avg_dwell_impact']:.2f} seconds")
    
    # Create and show dashboard
    dashboard = predictor.create_visualization_dashboard(
        data.iloc[len(data)-len(predictions):].reset_index(drop=True),
        y_test,
        predictions
    )
    dashboard.show()

if __name__ == "__main__":
    main()