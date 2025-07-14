# --- app.py --- 
from flask import Flask, render_template, request
import numpy as np
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained models and preprocessing tools
classifier_model = joblib.load(open('rf_model.joblib', 'rb'))  # Classifier for scenario 1
scaler_cls = joblib.load(open('scaler.joblib', 'rb'))   # Scaler used for scenario 1
label_encoder = joblib.load(open('encoder.joblib', 'rb'))  # LabelEncoder for scenario 1

regression_model = joblib.load(open('redshift_model.joblib', 'rb'))  # Regressor for scenario 2
scaler_reg = joblib.load(open('scaler_redshift.joblib', 'rb'))  # Scaler for scenario 2

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Classification Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form.get('g')),
            float(request.form.get('r')),
            float(request.form.get('i')),
            float(request.form.get('z')),
            float(request.form.get('petroR50_u')),
            float(request.form.get('petroR50_g')),
            float(request.form.get('psfMag_i')),
            float(request.form.get('psfMag_z')),
        ]
        input_array = np.array([features])
        scaled_input = scaler_cls.transform(input_array)
        encoded_prediction = classifier_model.predict(scaled_input)
        
        # Fix: Reshape the 1D prediction to 2D for inverse_transform
        decoded_prediction = label_encoder.inverse_transform(encoded_prediction.reshape(-1, 1))[0]
        
        return render_template('inner-page.html', result=f"Star Formation Type: {decoded_prediction}")

    except Exception as e:
        return render_template('inner-page.html', result=f"Error: {str(e)}")

# Redshift Regression Route
@app.route('/redshift', methods=['POST'])
def redshift():
    try:
        features = [
            float(request.form.get('u')),
            float(request.form.get('g')),
            float(request.form.get('r')),
            float(request.form.get('i')),
            float(request.form.get('z')),
            float(request.form.get('petroR50_i')),
            float(request.form.get('petroR50_r')),
            float(request.form.get('psfMag_u')),
            float(request.form.get('psfMag_r')),
            float(request.form.get('psfMag_g')),
        ]
        input_array = np.array([features])
        scaled_input = scaler_reg.transform(input_array)
        prediction = regression_model.predict(scaled_input)[0]
        return render_template('inner-page.html', result=f"Predicted Redshift: {prediction:.5f}")

    except Exception as e:
        return render_template('inner-page.html', result=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)