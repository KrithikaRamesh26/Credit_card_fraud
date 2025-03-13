from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

# Load trained XGBoost model
xgb_model = XGBClassifier()
xgb_model.load_model("xgb_new.json")  # Save and load XGBoost model

# Define feature columns
feature_columns = ['cc_num', 'merchant', 'category', 'amt', 'gender', 'lat', 'long', 
                   'city_pop', 'job', 'merch_lat', 'merch_long', 'Collision_score', 'hour', 'day', 'month']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from request
        df = pd.DataFrame([data], columns=feature_columns)  # Convert to DataFrame

        # Ensure correct data type
        df = df.astype(float)

        # Make prediction with XGBoost model and round to 4 decimal places
        xgb_pred = [round(prob, 4) for prob in xgb_model.predict_proba(df)[:, 1].tolist()]

        response = {
            "XGBoost_Prediction": xgb_pred
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
