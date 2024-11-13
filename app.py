import os
import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 
# 제목 설정
st.title("서울시의 집값 예측")

# 모델 불러오기
with open("xgboost_feature3 (1).pkl", "rb") as file:
    model = pickle.load(file)
    
 
# 예시 데이터 불러오기
data = pd.read_csv("seoul_housing_data_2024.csv")

# 접수연도를 정수형으로 변환
data['접수연도'] = pd.to_numeric(data['접수연도'], errors='coerce').fillna(0).astype(int)

# 사용자 선택 UI 생성
districts = data['자치구명'].unique()
years = data['접수연도'].unique()

selected_district = st.selectbox("자치구 선택", districts)
selected_year = st.selectbox("접수연도 선택", sorted(years))

# 선택한 구와 연도에 따른 데이터 필터링
district_data = data[(data['자치구명'] == selected_district) & (data['접수연도'] == selected_year)]

# 예측 수행 및 출력
if not district_data.empty:
    columns_to_drop = ['계약일', '자치구명']
    X = district_data.drop(columns=columns_to_drop, errors='ignore')

    # 범주형 변수 인코딩
    categorical_columns = ['법정동명', '건물용도']
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # 모든 값을 숫자형으로 변환하고 NaN을 0으로 채움
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 모델의 입력 피처 수에 맞추기
    required_features = model.get_booster().feature_names
    for feature in required_features:
        if feature not in X.columns:
            X[feature] = 0
    X = X[required_features]

    # 예측 수행 후 법정동명을 인덱스로 설정
    district_data['예측가격'] = model.predict(X.values)
    
    # 법정동명별 평균 예측가격 계산
    average_price = district_data.groupby('법정동명')['예측가격'].mean().reset_index()

    # 예측된 가격을 데이터프레임으로 표시 (법정동명 열을 인덱스로)
    average_price.set_index('법정동명', inplace=True)
    st.write(f"**{selected_district}의 {selected_year}년 예측 주택 가격 (법정동명별 평균)**")
    st.dataframe(average_price[['예측가격']], use_container_width=True)

    # 한글 폰트 설정 (한글 깨짐 해결)
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # Linux 예시
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows의 기본 한글 폰트 경로
    font_prop = font_manager.FontProperties(fname=font_path)

    # 예측된 가격에 대한 그래프 표시 (법정동명별 평균 가격)
    plt.figure(figsize=(10, 6))
    st.write("**법정동명별 예측 주택 가격 그래프(평균)**")
    average_price['예측가격'].plot(kind='bar', color='skyblue', title=f'{selected_district} {selected_year} 법정동명별 평균 예측 주택 가격', fontsize=12)
    plt.xlabel('법정동명', fontsize=12)
    plt.ylabel('평균 예측 가격 (단위:만원)', fontsize=12)  # 단위 추가

    # 그래프에 한글 폰트 설정
    plt.title(f'{selected_district} {selected_year} 법정동명별 평균 예측 주택 가격(평균)', fontproperties=font_prop)
    plt.xlabel('법정동명', fontproperties=font_prop)
    plt.ylabel('평균 예측 가격 (단위:만원)', fontproperties=font_prop)  # 단위 추가
    plt.xticks(fontproperties=font_prop)

    st.pyplot(plt)

else:
    st.write("선택한 구와 연도의 데이터가 없습니다.")
