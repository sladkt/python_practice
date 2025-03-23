from flask import Flask, jsonify, render_template
import predict_weather
import requests
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/predict')
def predict():
    model_prediction = predict_weather.predicted_rf()   # 랜덤 포레스트 예측 값
    
    LAT, LON = 37.5665, 126.9780
    future_api = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "temperature_2m_max",
        "timezone": "Asia/Seoul"
    }
    response = requests.get(future_api, params=params)
    forecast = response.json()
    real_forecast = forecast['daily']['temperature_2m_max'][1]   # 실제 예보

    return render_template('index.html', model_prediction=model_prediction, real_forecast=real_forecast) 

if __name__ == '__main__':
    app.run(debug=True)