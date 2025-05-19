# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:56:36 2025

@author: gusdk
"""

import csv
import matplotlib.pyplot as plt

# 파일 열기
f1 = open('subwaytime_201803.csv', encoding='utf-8')
f2 = open('subwaytime_202003.csv', encoding='utf-8')
f3 = open('subwaytime_202503.csv', encoding='utf-8')

data1 = csv.reader(f1)
data2 = csv.reader(f2)
data3 = csv.reader(f3)

# 헤더 건너뛰기
next(data1)
next(data2)
next(data3)

# 각 데이터의 시간대별 승차/하차 인원 초기화
s_in_2018 = [0] * 24
s_out_2018 = [0] * 24

s_in_2020 = [0] * 24
s_out_2020 = [0] * 24

s_in_2025 = [0] * 24
s_out_2025 = [0] * 24

# 데이터를 처리하는 함수
def process_data(data, s_in, s_out):
    for row in data:
        row[4:] = map(lambda x: int(x.replace(',', '')), row[4:])  # 쉼표 제거 후 숫자로 변환
        for i in range(24):
            s_in[i] += row[4 + i * 2]   # 시간대별 승차인원
            s_out[i] += row[5 + i * 2]  # 시간대별 하차인원

# 각 데이터 처리
process_data(data1, s_in_2018, s_out_2018)
process_data(data2, s_in_2020, s_out_2020)
process_data(data3, s_in_2025, s_out_2025)

# 그래프 그리기
plt.figure(dpi=300)
plt.rc('font', family='Malgun Gothic')
plt.title('지하철 시간대별 승하차 인원 추이')

# 승차 인원 그래프
plt.plot(s_in_2018, label='2018년 3월 승차')
plt.plot(s_in_2020, label='2020년 3월 승차')
plt.plot(s_in_2025, label='2025년 3월 승차')

# 하차 인원 그래프 (선택적으로 추가 가능)
plt.plot(s_out_2018, '--', label='2018년 3월 하차')
plt.plot(s_out_2020, '--', label='2020년 3월 하차')
plt.plot(s_out_2025, '--', label='2025년 3월 하차')

# 그래프 설정
plt.legend()
plt.xticks(range(24), range(4, 28))  # X축 시간대 설정 (4시부터 시작)
plt.xlabel('시간대')
plt.ylabel('승하차 인원 (단위: 명)')
plt.show()
