# -*- coding: utf-8 -*-
"""
빅데이터 - 4주 차 과제 - 컴퓨터공학전공 20210905 최현지 
"""

# (1) 201502_population.csv 파일을 읽어오기 (2021502_주민등록인구및세대현황.csv의 이름을 임의로 바꿈) 

"""
(2) 데이터 처리
csv 파일에서 17개 지역별 인구 수를 읽어서 인구 변동 수를 계산한 후 배열에 저장한다.
인구 수 를 처리할 때 콤마(,)를빼고 숫자로 계산한다. (replace 함수사용)
"""
import pandas as pd

# 1) 2015 2월 데이터 수집 후 불러오기 
df_2015 = pd.read_csv("201502_population.csv",  encoding='ANSI')

# 엥ㅋㅋ csv보다 Pandas가 최고네 !! ㅋㅋㅋ

# 콤마(,)를 빼고 숫자로 변환 
df_2015['2015년02월_총인구수'] =df_2015['2015년02월_총인구수'].str.replace(',', '').astype(int)

# 2) 2025 2월 데이터 수집 후 불러오기 
df_2025 = pd.read_csv("202502_population.csv", encoding="ANSI")

df_2025['2025년02월_총인구수'] = df_2025['2025년02월_총인구수'].str.replace(',', '').astype(int)

# 독립적인 두 데이터프레임의 경우, pd.merge()를 통해 병합하는 게 안전하지만, 
# 해당 두 프레임은 행정구역의 순서가 동일하므로, 
# 기존 dataFrame에 새로운 행을 추가하여, 단지 두 프레임의 값을 빼주는 것으로 그치겠다. 
df_2025['인구변동'] = df_2015['2015년02월_총인구수'].values - df_2025['2025년02월_총인구수'].values 


# 인구 변동 값을 -1.5에서 1.0 사이로 정규화
min_val = df_2025['인구변동'].min()
max_val = df_2025['인구변동'].max()
df_2025['정규화_인구변동'] = -1.5 + ((df_2025['인구변동'] - min_val) * (1.0 - (-1.5))) / (max_val - min_val)

"""
(3) bar 차트를 그린다.
"""


import matplotlib.pyplot as plt
# import koreanize_matplotlib


"""
# 한글 폰트 출력하기 위해 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'NanumGothic', 'AppleGothic' 등
plt.rcParams['axes.unicode_minus'] = False 


plt.rc('font', family = 'Malgun Gothic') # 맑은 고딕을 기본 글꼴로 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

★★★★★ 다 필요없고 
plt.rc('font', family ='Malgun Gothic')  
★★★★★ 이 코드가 짱임 ↑↑↑
"""

plt.rc('font', family ='Malgun Gothic')  
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# '행정구역' 열에서 괄호와 숫자를 제거
df_2025['행정구역'] = df_2025['행정구역'].str.extract(r'^(.*?)(?=\s\()')

# 값에 따라 색상 결정 (양수: 하늘색, 음수: 연한 빨간색)
colors = ['skyblue' if x > 0 else 'lightcoral' for x in df_2025['정규화_인구변동']]

# 막대 그래프 그리기
plt.figure(figsize=(12, 6)) # 그래프의 가로, 세로 크기를 정한다.
bars = plt.bar(df_2025['행정구역'], df_2025['정규화_인구변동'], color=colors)

# 그래프 제목 및 축 레이블 추가
plt.title('2015-2025 지역별 인구 변동', fontsize=16)
plt.xlabel('행정구역', fontsize=12)
plt.ylabel('인구 변동량', fontsize=12)

# X축 레이블 회전 및 레이아웃 조정
plt.xticks(rotation=90, fontsize=10) # rotation으로 수직 정렬, 45도 정도가 비스듬히 보여서 가독성 좋긴 함
plt.tight_layout()

# Y축 값 텍스트로 표시 (양수는 막대 위, 음수는 막대 아래)

#그래프 수치 표현
for bar in bars:
    height = bar.get_height()
    height_text = f'{height:.2f}'  # 소수점 둘째 자리까지 포맷팅
    plt.text(bar.get_x()+bar.get_width()/2.0, height + 0.02, height_text,ha = 'center',va='bottom',size=12, color='red')


# 그래프 저장
plt.savefig('population_change_bar_chart.png')

plt.rc('font', family ='Malgun Gothic')       #한글폰트사용하기

# 그래프 출력
plt.show()









"""


각 지역별 인구변동을 2015년 2월, 2025년 2월에 대하여 그래프로 그려보시요.
즉Q4.1 문제를pandas 로읽어서처리해본다


"""