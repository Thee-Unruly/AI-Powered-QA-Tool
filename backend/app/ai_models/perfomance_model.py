import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

# Load your historical performance data (assuming you have data on CPU, memory usage, etc.)
def load_data():
    # Example of loading a CSV file, replace with your actual data
    data = pd.read_csv("performance_data.csv")
    return data

# Data Preprocessing and Feature Engineering
def preprocess_data(data):
    # Feature engineering: select relevant columns
    features = data[['cpu_usage', 'memory_usage', 'disk_io', 'network_load']]  # Example features
    labels = data['performance_label']  # Binary label indicating high/low performance
    
    # Scaling features (optional but helps some models perform better)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels, scaler

# Train the Model
def train_model(features, labels):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Using Random Forest as the model (you can experiment with others like XGBoost or SVM)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Hyperparameter tuning using GridSearchCV (optional)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Model evaluation
    y_pred = best_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    
    # Saving the model for future use (optional)
    joblib.dump(best_model, 'performance_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as well
    
    return best_model

# Load the trained model and make predictions
def predict_performance(model, scaler, new_data):
    # Ensure new data is scaled properly
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    
    return prediction

# Main function to run the process
if __name__ == "__main__":
    # Step 1: Load and preprocess data
    data = load_data()
    features, labels, scaler = preprocess_data(data)
    
    # Step 2: Train model
    model = train_model(features, labels)
    
    # Step 3: Predict performance on new data (you can input new system performance metrics here)
    new_data = np.array([[80, 70, 50, 20]])  # Example new data: [cpu_usage, memory_usage, disk_io, network_load]
    prediction = predict_performance(model, scaler, new_data)
    print(f"Predicted Performance Label: {prediction[0]}")  # 1 for high performance, 0 for low performance
