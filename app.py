from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ✅ Home route so browser doesn't show 404
@app.route('/', methods=['GET'])
def home():
    return "✅ Flask classifier server is running."
# Load model and feature list
model = joblib.load('trained_data/model_cls.pkl')
features = joblib.load('trained_data/model_features.pkl')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input JSON as DataFrame
    input_data = pd.DataFrame([request.json])
    
    # One-hot encode and reindex to match training features
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    # Predict
    pred = model.predict(input_encoded)[0]
    label = "Fail" if pred == 1 else "Pass"

    return jsonify({"Prediction": label})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
