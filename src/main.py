# station_dwell_predictor/src/main.py

from models.predictor import StationDwellPredictor
from utils.data_generator import generate_dummy_data
from visualization.dashboard import create_visualization_dashboard
from sklearn.model_selection import train_test_split

#https://github.com/Umaraslam66/station-dwell-predictor
# git remote add origin https://github.com/Umaraslam66/station-dwell-predictor.git
# git branch -M main
# git push -u origin main




def main():
    # Initialize predictor
    predictor = StationDwellPredictor()
    
    # Generate and prepare data
    print("Generating synthetic data...")
    data = generate_dummy_data(n_samples=5000)
    
    # Split features and target
    X = data.drop('dwell_time', axis=1)
    y = data['dwell_time']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining models...")
    best_model, cv_score = predictor.train(X_train, y_train)
    print(f"Best model: {best_model} (CV Score: {cv_score:.3f})")
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Create and show dashboard
    dashboard = create_visualization_dashboard(
        data=data,
        true_values=y_test,
        predictions=predictions,
        feature_names=X.columns,
        model=predictor.best_model
    )
    dashboard.show()

if __name__ == "__main__":
    main()