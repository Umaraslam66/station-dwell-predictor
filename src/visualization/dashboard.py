import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_dashboard(data, true_values, predictions, feature_importance):
    """Create a comprehensive visualization dashboard"""
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
    fig.add_trace(
        go.Bar(x=list(feature_importance.keys()), 
               y=list(feature_importance.values()),
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