# 🌤️ Weather Prediction using XGBoost & Flask

### 📌 프로젝트 개요
이 프로젝트는 **데이터 과학 입문 실습 프로젝트**로,  
**과거 3년 간의 기상 데이터**를 활용하여 **내일의 최고 기온을 예측**합니다.

- 활용 데이터: 온도, 습도, 기압, 풍속, 구름량, 강수량
- 사용 모델: **Linear Regression**, **Polynomial Regression**, **Random Forest**, **XGBoost**
- 모델 평가 및 특성 중요도 분석 포함

예측된 결과는 Flask 웹 애플리케이션을 통해 실시간으로 확인할 수 있으며,  
**Open-Meteo API의 실제 예보**와 함께 비교하여 확인할 수 있습니다.

---

### 🛠️ 사용 기술
- **Python**
- **Flask**
- **pandas**
- **scikit-learn**
- **XGBoost**
- **Open-Meteo API**
- **GitHub**

---

### 💡 주요 기능
- ✅ 다양한 머신러닝 모델을 통한 **내일 최고 기온 예측**
- ✅ **실시간 Open-Meteo 예보 데이터** 수집
- ✅ 모델 평가 (MSE, R²) 및 **특성 중요도(Feature Importance)** 시각화
- ✅ Flask API를 통해 웹에서 **예측 결과 & 실측 예보 비교** 제공
- ✅ HTML을 활용한 예쁜 웹 페이지 출력

---

### 🚀 실행 방법
```bash
# 패키지 설치
pip install -r requirements.txt

# 서버 실행
python app.py
실행 후 브라우저에서 http://localhost:5000/predict 또는 메인 페이지로 접속

---

### 📊 향후 개선 아이디어
1. 예측 결과의 시각화 강화 (차트 추가)
2. 날짜 선택 기능으로 과거 데이터 예측 비교
3. 매일 예측 결과 자동 기록 및 시계열 분석

---

### 🙌 개발자의 한마디

데이터 과학에 대한 흥미를 실전으로 옮겨본 첫 프로젝트입니다.
이론적인 내용을 글이 아닌 직접 모델을 만들어 예측, 평가, 개선하고 웹에 연결해보는
경험으로 머신러닝의 실용성과 데이터 과학의 흥미를 더욱 느낄 수 있었던 프로젝트였습니다.

---