"""
Created on Sat Nov 8 2024

@author: Rohit kumar
"""

from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder="template")
model = pickle.load(open(r"C:\Users\Rohit Kumar\Downloads\Rainfall-Prediction-main\Rainfall-Prediction-main\xg_random.pkl", "rb"))
print("Model Loaded")

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

def safe_float_conversion(value, default=0.0):
    """Convert value to float, return default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # DATE
        date = request.form['date']
        day = safe_float_conversion(pd.to_datetime(date, format="%Y-%m-%d").day)
        month = safe_float_conversion(pd.to_datetime(date, format="%Y-%m-%d").month)

        # Collecting other inputs with safe conversion
        minTemp = safe_float_conversion(request.form['mintemp'])
        maxTemp = safe_float_conversion(request.form['maxtemp'])
        rainfall = safe_float_conversion(request.form['rainfall'])
        evaporation = safe_float_conversion(request.form['evaporation'])
        sunshine = safe_float_conversion(request.form['sunshine'])
        windGustSpeed = safe_float_conversion(request.form['windgustspeed'])
        windSpeed9am = safe_float_conversion(request.form['windspeed9am'])
        windSpeed3pm = safe_float_conversion(request.form['windspeed3pm'])
        humidity9am = safe_float_conversion(request.form['humidity9am'])
        humidity3pm = safe_float_conversion(request.form['humidity3pm'])
        pressure9am = safe_float_conversion(request.form['pressure9am'])
        pressure3pm = safe_float_conversion(request.form['pressure3pm'])
        temp9am = safe_float_conversion(request.form['temp9am'])
        temp3pm = safe_float_conversion(request.form['temp3pm'])
        cloud9am = safe_float_conversion(request.form['cloud9am'])
        cloud3pm = safe_float_conversion(request.form['cloud3pm'])
        location = safe_float_conversion(request.form['location'])
        winddDir9am = safe_float_conversion(request.form['winddir9am'])
        winddDir3pm = safe_float_conversion(request.form['winddir3pm'])
        windGustDir = safe_float_conversion(request.form['windgustdir'])
        rainToday = safe_float_conversion(request.form['raintoday'])

        # Create input list for model prediction
        input_lst = [location, minTemp, maxTemp, rainfall, evaporation, sunshine,
                    windGustDir, windGustSpeed, winddDir9am, winddDir3pm,
                    windSpeed9am, windSpeed3pm, humidity9am, humidity3pm,
                    pressure9am, pressure3pm, cloud9am, cloud3pm,
                    temp9am, temp3pm, rainToday, month, day]

        # Reshape input for prediction
        input_array = np.array(input_lst).reshape(1, -1)  # Reshape to 2D array

        # Make the prediction
        pred = model.predict(input_array)
        output = pred[0]  # Get the first element of the prediction

        # Render templates based on the prediction
        if output == 0:
            return render_template("sunny.html")
        else:
            return render_template("rainy.html")

    # Fallback for GET requests or if something goes wrong
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)