import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import GridSearchCV
import joblib

# Load your dataset (e.g., security-related logs, code snippets, configurations)
def load_data():
    # Example loading of a CSV file containing code snippets and labels
    data = pd.read_csv("security_logs.csv")
    return data

# Data Preprocessing and Feature Engineering
def preprocess_data(data):
    # For simplicity, assume data has 'code' (for code snippets or logs) and 'vulnerable' (target label)
    # Example: ['code', 'vulnerable']
    
    # Convert textual data into numerical features using TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(data['code']).toarray()  # Convert 'code' into features
    y = LabelEncoder().fit_transform(data['vulnerable'])  # Label encode the target variable
    
    return X, y, tfidf_vectorizer

# Model Training - Random Forest, XGBoost, and LSTM for text classification
def train_models(X, y):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}

    # RandomForest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # LSTM (for sequence data like code logs over time, if applicable)
    # Reshape the input data for LSTM model
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
            joblib.dump(model, f'{model_name}_security_model.pkl')
        else:
            model.save('LSTM_security_model.h5')
    
    return models

# Anomaly Detection for Configuration Files or Logs
def anomaly_detection(X):
    # Using Isolation Forest for anomaly detection in system configurations or logs
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% anomalies expected
    anomaly_predictions = isolation_forest.fit_predict(X)
    
    # -1 indicates anomalies, 1 indicates normal points
    anomalies = np.where(anomaly_predictions == -1)
    print(f"Anomalies detected: {len(anomalies[0])} out of {len(X)} samples")
    
    return anomalies

# Load the trained model and make predictions
def predict_vulnerability(model, tfidf_vectorizer, new_data):
    # Transform new data using the same TF-IDF vectorizer
    new_data_tfidf = tfidf_vectorizer.transform(new_data).toarray()
    prediction = model.predict(new_data_tfidf)
    
    return prediction

# Main function to run the process
if __name__ == "__main__":
    # Step 1: Load and preprocess data
    data = load_data()
    X, y, tfidf_vectorizer = preprocess_data(data)
    
    # Step 2: Train models
    models = train_models(X, y)
    
    # Step 3: Anomaly detection (for system configurations or logs)
    anomalies = anomaly_detection(X)
    
    # Step 4: Predict security vulnerability on new data
    new_data = ["if (user_input == 'admin') { execute_admin_command(); }"]  # Example of a potential vulnerability
    rf_model = models['RandomForest']
    prediction = predict_vulnerability(rf_model, tfidf_vectorizer, new_data)
    print(f"Predicted Vulnerability (RandomForest): {prediction[0]}")  # 1 for vulnerable, 0 for non-vulnerable
