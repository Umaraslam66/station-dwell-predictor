import numpy as np
import pandas as pd

def generate_dummy_data(n_samples=1000):
    """Generate synthetic station data with realistic patterns"""
    np.random.seed(42)
    
    # Time-based features
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    hours = dates.hour
    
    # Generate passenger volume with realistic patterns
    passenger_volume = np.zeros(n_samples)
    for i, hour in enumerate(hours):
        if 7 <= hour <= 9:  # Morning peak
            passenger_volume[i] = np.random.normal(800, 150)
        elif 16 <= hour <= 18:  # Evening peak
            passenger_volume[i] = np.random.normal(900, 200)
        else:  # Normal hours
            passenger_volume[i] = np.random.normal(400, 100)

    # Generate other features
    platform_length = np.random.choice([100, 150, 200, 250], n_samples)
    peak_hour = np.where((hours >= 7) & (hours <= 9) | 
                        (hours >= 16) & (hours <= 18), 1, 0)
    
    # Weather conditions
    month = dates.month
    weather_condition = np.zeros(n_samples)
    
    # Winter months
    winter_mask = month.isin([12, 1, 2])
    weather_condition[winter_mask] = np.random.choice([0, 1, 2], 
                                                    size=winter_mask.sum(), 
                                                    p=[0.4, 0.3, 0.3])
    
    # Spring/Fall months
    spring_fall_mask = month.isin([3, 4, 5, 9, 10, 11])
    weather_condition[spring_fall_mask] = np.random.choice([0, 1, 2], 
                                                         size=spring_fall_mask.sum(), 
                                                         p=[0.6, 0.3, 0.1])
    
    # Summer months
    summer_mask = month.isin([6, 7, 8])
    weather_condition[summer_mask] = np.random.choice([0, 1, 2], 
                                                    size=summer_mask.sum(), 
                                                    p=[0.8, 0.2, 0.0])

    # Additional features
    special_event = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    station_complexity = np.random.choice([1, 2, 3], n_samples)
    
    # Generate target (dwell time)
    base_dwell_time = 60
    dwell_time = (
        base_dwell_time +
        passenger_volume * 0.1 +
        peak_hour * 30 +
        weather_condition * 15 +
        special_event * 45 +
        station_complexity * 10 +
        np.random.normal(0, 10, n_samples)
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
    
    return data