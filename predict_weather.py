import numpy as np
import requests
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# 서울의 위도, 경도
LAT, LON = 37.5665, 126.9780  
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 날짜 계산 (최근 30일)
today = datetime.date.today()
start_date = today - datetime.timedelta(days=30)
end_date = today - datetime.timedelta(days=2)

# API 요청
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "daily": "temperature_2m_max,relative_humidity_2m_max,wind_speed_10m_max,surface_pressure_max,precipitation_sum,cloud_cover_mean",
    "timezone": "Asia/Seoul"
}

response = requests.get(BASE_URL, params=params)
data = response.json()

# 데이터 검증
if "daily" not in data:
    print("데이터를 가져오는 데 실패했습니다.")
    exit()

# 데이터 추출
past_temps = data["daily"]["temperature_2m_max"]
humidity = data["daily"]["relative_humidity_2m_max"]
wind_speed = data["daily"]["wind_speed_10m_max"]
pressure = data["daily"]["surface_pressure_max"]
precipitation = data["daily"]["precipitation_sum"]
cloud_cover = data["daily"]["cloud_cover_mean"]

days = np.array(range(1, len(past_temps) + 1)).reshape(-1, 1)
temps = np.array(past_temps).reshape(-1, 1)
features = np.hstack((
    np.array(humidity).reshape(-1, 1),
    np.array(wind_speed).reshape(-1, 1),
    np.array(pressure).reshape(-1, 1),
    np.array(precipitation).reshape(-1, 1),
    np.array(cloud_cover).reshape(-1, 1)
))

# 선형 회귀 모델 학습
linear_model = LinearRegression()
linear_model.fit(days, temps)

multi_feature_model = LinearRegression()
multi_feature_model.fit(features, temps)

# 다항 회귀 모델 학습 (2차)
poly = PolynomialFeatures(degree=2)
days_poly = poly.fit_transform(days)
poly_model = LinearRegression()
poly_model.fit(days_poly, temps)

# 내일 예측
tomorrow_day = np.array([[len(past_temps) + 1]])
predicted_temp_linear = linear_model.predict(tomorrow_day)[0][0]
predicted_temp_poly = poly_model.predict(poly.transform(tomorrow_day))[0][0]
predicted_temp_multi = multi_feature_model.predict(features[-1].reshape(1, -1))[0][0]

print(f"\n🌡️ 선형 회귀 예측: {predicted_temp_linear:.2f}°C")
print(f"🌡️ 다항 회귀 예측: {predicted_temp_poly:.2f}°C")
print(f"🌡️ 다중 특성 회귀 예측: {predicted_temp_multi:.2f}°C")

# 모델 평가
mse_linear = mean_squared_error(temps, linear_model.predict(days))
r2_linear = r2_score(temps, linear_model.predict(days))

mse_poly = mean_squared_error(temps, poly_model.predict(days_poly))
r2_poly = r2_score(temps, poly_model.predict(days_poly))

mse_multi = mean_squared_error(temps, multi_feature_model.predict(features))
r2_multi = r2_score(temps, multi_feature_model.predict(features))

print(f"\nLinear Regression MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}")
print(f"Polynomial Regression MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")
print(f"Multiple Feature Regression MSE: {mse_multi:.2f}, R²: {r2_multi:.2f}")

# 그래프 시각화
plt.scatter(days, temps, color='blue', label='Actual Temperature')
plt.plot(days, linear_model.predict(days), color='red', label='Linear Regression')
plt.plot(days, poly_model.predict(days_poly), color='purple', label='Polynomial Regression')
plt.scatter(tomorrow_day, predicted_temp_linear, color='green', label=f'Tomorrow (Linear): {predicted_temp_linear:.2f}°C')
plt.scatter(tomorrow_day, predicted_temp_poly, color='yellow', label=f'Tomorrow (Poly): {predicted_temp_poly:.2f}°C')
plt.scatter(tomorrow_day, predicted_temp_multi, color='black', label=f'Tomorrow (Multi): {predicted_temp_multi:.2f}°C')

plt.title('Temperature Prediction')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('temperature_prediction.png', bbox_inches='tight', dpi=300)
plt.show()
