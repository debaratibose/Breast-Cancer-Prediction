from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
# Load model, scaler, and selected features
model = joblib.load("breast_cancer_model.joblib")
scaler = joblib.load("scaler.joblib")
selected_features = joblib.load("selected_features.joblib")  # Load the top 10 feature names

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', feature_names=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get only selected features from form input
        features = [float(request.form[feature]) for feature in selected_features]

        # Convert to DataFrame with correct feature ordering
        input_data = pd.DataFrame([features], columns=selected_features)

        # Scale input data
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        return render_template('index.html', prediction_text=f'The tumor is likely: {result}', feature_names=selected_features)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', feature_names=selected_features)

if __name__ == '__main__':
    app.run(debug=True)
