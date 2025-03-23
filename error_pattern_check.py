import pandas as pd
import datetime
import predict_weather

def save_to_csv(predicted_temp, real_temp):

    # 날짜 구하기
    today = datetime.date.today()
    date = today.strftime("%Y-%m-%d")

    # 데이터 저장할 리스트
    data = {
        "Date": [date],
        "predicted_temp": [predicted_temp],
        "Real_temp": [real_temp],
        "Error": round([predicted_temp - real_temp], 2)
    }

    # DataFrame으로 변환
    df = pd.DataFrame(data)

    # 파일에 저장
    df.to_csv('predictions.csv', mode='a', header=not pd.io.common.file_exists('predictions.csv'), index=False)

    
predicted_temp = predict_weather.predicted_rf()
real_temp = predict_weather.get_real_temp()
save_to_csv(predicted_temp, real_temp)