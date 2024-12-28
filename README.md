# Station Dwell Time Predictor

A machine learning model to predict station dwell times in public transportation systems.

## Features
- Multiple model support (Random Forest, Gradient Boosting)
- Advanced data generation
- Anomaly detection
- Comprehensive visualization dashboard
- Alert system

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.models.predictor import StationDwellPredictor

predictor = StationDwellPredictor()
data = predictor.generate_dummy_data()
# ... continue with your analysis
```

## License
MIT
