import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# AWS Elastic Beanstalk looks specifically for the 'application' variable
application = Flask(__name__)
app = application

## Import ridge regressor and standard scaler pickle
# Using 'with' statement is safer for opening files
with open('models/ridge.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extracting data from form and converting to float
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scaling and Prediction
            # Note: Ensure features are in the exact same order as used during training
            features = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            new_data_scaled = standard_scaler.transform(features)
            result = ridge_model.predict(new_data_scaled)

            # Pass result[0] to the template
            return render_template('home.html', result=result[0])
            
        except Exception as e:
            # If data conversion fails or something goes wrong, return the error
            return render_template('home.html', result=f"Error: {str(e)}")
    
    else:
        # CRITICAL: Pass result=None so the template knows no prediction has happened yet
        return render_template('home.html', result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0")