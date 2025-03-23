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
    "daily": ["temperature_2m_max"],
    "timezone": "Asia/Seoul"
}

response = requests.get(BASE_URL, params=params)
data = response.json()

# 데이터 추출 및 출력
if "daily" in data:
    past_temps = data["daily"]["temperature_2m_max"]
    days = np.array(range(1, len(past_temps) + 1)).reshape(-1, 1) # 날짜 1, 2, 3, ...
    temps = np.array(past_temps).reshape(-1, 1) # 기온 데이터

    print("\n📊 최근 30일간의 데이터 (X: 날짜, Y: 기온)")
    for i in range(len(past_temps)):
        print(f"Day {i+1}: {past_temps[i]}C")

    # 선형 회귀 모델 생성
    model_linear = LinearRegression()
    model_linear.fit(days, temps)  # 선형 회귀 모델 학습

    # 다항 회귀 모델 (차수 2)
    poly = PolynomialFeatures(degree=2)
    days_poly = poly.fit_transform(days)  # 다항식으로 변환
    model_poly = LinearRegression()
    model_poly.fit(days_poly, temps)  # 다항 회귀 모델 학습

    # 내일 예측 (tomorrow)
    tomorrow_day = len(past_temps) + 1
    tomorrow = np.array([[tomorrow_day]])

    predicted_temp_linear = model_linear.predict(tomorrow)[0][0]  # 선형 회귀 예측
    predicted_temp_poly = model_poly.predict(poly.transform(tomorrow))[0][0]  # 다항 회귀 예측

    print(f"\n🌡️ 예측된 내일 최고 기온 (선형 회귀): {predicted_temp_linear:.2f}°C")
    print(f"🌡️ 예측된 내일 최고 기온 (다항 회귀): {predicted_temp_poly:.2f}°C")

else:
    print("데이터를 가져오는 데 실패했습니다.")

# 예측값 (predicted_temp_linear)과 실제값 (temps) 비교
predicted_temps_linear = model_linear.predict(days)
predicted_temps_poly = model_poly.predict(days_poly)

# MSE (Mean Squared Error) 계산
mse_linear = mean_squared_error(temps, predicted_temps_linear)
mse_poly = mean_squared_error(temps, predicted_temps_poly)

# R^2 (결정계수) 계산
r2_linear = r2_score(temps, predicted_temps_linear)
r2_poly = r2_score(temps, predicted_temps_poly)

print(f"\nLinear Regression MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}")
print(f"Polynomial Regression MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")

# 과거 데이터 시각화
plt.scatter(days, temps, color='blue', label='Actual Temperature')  # 실제 기온 (scatter로 표시)

# 예측된 값 시각화 (선형 회귀 결과)
plt.plot(days, predicted_temps_linear, color='red', label='Linear Regression Model')  # 선형 회귀 예측

# 다항 회귀 모델 값 시각화
plt.plot(days, predicted_temps_poly, color='purple', label='Poly Regression Model')  # 다항 회귀 예측

# 내일 예측된 기온 표시 (선형 회귀)
plt.scatter(tomorrow_day, predicted_temp_linear, color='green', zorder=5, label=f'Predicted Tomorrow Temp (linear): {predicted_temp_linear:.2f}°C')

# 내일 예측된 기온 표시 (다항 회귀)
plt.scatter(tomorrow_day, predicted_temp_poly, color='yellow', zorder=5, label=f'Predicted Tomorrow Temp (poly): {predicted_temp_poly:.2f}°C')

# 그래프 제목 및 레이블
plt.title('Predicted Maximum Temperatures for One Month')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.legend()

# 범례를 그래프 밖으로 배치
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 범례를 그래프의 오른쪽 상단으로 이동

# 그래프를 파일로 저장
plt.savefig('temperature_prediction.png', bbox_inches='tight')  # bbox_inches='tight'로 여백 조정
plt.show()  # 그래프 표시
