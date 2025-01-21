import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

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

# Model Training - RandomForest, XGBoost, SVM, and LSTM
def train_models(features, labels):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    models = {}

    # RandomForest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # SVM
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model
    
    # LSTM (Assuming time-series data, reshaping input for LSTM)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Reshape for LSTM
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
    models['LSTM'] = lstm_model

    # Evaluate the models
    for model_name, model in models.items():
        if model_name != 'LSTM':
            y_pred = model.predict(X_test)
            print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
            print(f"{model_name} Accuracy Score:", accuracy_score(y_test, y_pred))
        else:
            y_pred_lstm = model.predict(X_test_lstm)
            y_pred_lstm = (y_pred_lstm > 0.5).astype(int)  # Convert probabilities to binary label
            print(f"LSTM Classification Report:\n", classification_report(y_test, y_pred_lstm))
            print(f"LSTM Accuracy Score:", accuracy_score(y_test, y_pred_lstm))

    # Saving models for future use
    for model_name, model in models.items():
        if model_name != 'LSTM':
            joblib.dump(model, f'{model_name}_performance_model.pkl')
        else:
            model.save('LSTM_performance_model.h5')
    
    return models

# Anomaly Detection using Isolation Forest
def anomaly_detection(features):
    # Using Isolation Forest to detect anomalies
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% expected anomalies
    anomaly_predictions = isolation_forest.fit_predict(features)
    
    # -1 indicates anomalies, 1 indicates normal points
    anomalies = np.where(anomaly_predictions == -1)
    print(f"Anomalies detected: {len(anomalies[0])} out of {len(features)} samples")
    
    return anomalies

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
    
    # Step 2: Train models
    models = train_models(features, labels)
    
    # Step 3: Anomaly detection
    anomalies = anomaly_detection(features)
    
    # Step 4: Predict performance on new data (you can input new system performance metrics here)
    new_data = np.array([[80, 70, 50, 20]])  # Example new data: [cpu_usage, memory_usage, disk_io, network_load]
    rf_model = models['RandomForest']
    prediction = predict_performance(rf_model, scaler, new_data)
    print(f"Predicted Performance (RandomForest): {prediction[0]}")  # 1 for high performance, 0 for low performance
