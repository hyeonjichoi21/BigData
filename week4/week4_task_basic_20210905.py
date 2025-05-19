# -*- coding: utf-8 -*-
"""
빅데이터 - 4주 차 과제 - 컴퓨터공학전공 20210905 최현지
pandas 안 쓰고, 그냥 csv 읽어서 불러오기 
"""

import csv
import matplotlib.pyplot as plt

# (1) CSV 파일 읽기 (Pandas 없이 구현)
def read_population_data(filename):
    data = []
    with open(filename, 'r', encoding='ANSI') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)  # 헤더 건너뛰기
        for row in csv_reader:
            region = row[0].split(' (')[0]  # 괄호 앞부분만 추출
            population = int(row[1].replace(',', ''))  # 콤마 제거 후 정수 변환
            data.append((region, population))
    return data

# 데이터 로드
data_2015 = read_population_data('201502_population.csv')
data_2025 = read_population_data('202502_population.csv')

# (2) 데이터 처리
regions = [d[0] for d in data_2015]
pop_2015 = [d[1] for d in data_2015]
pop_2025 = [d[1] for d in data_2025]

# 인구 변동 계산
changes = [p2015 - p2025 for p2015, p2025 in zip(pop_2015, pop_2025)]

# 정규화 계산 (-1.5 ~ 1.0 범위)
min_change = min(changes)
max_change = max(changes)
normalized = [-1.5 + (c - min_change) * 2.5 / (max_change - min_change) for c in changes]

# (3) 시각화
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))
colors = ['skyblue' if x > 0 else 'lightcoral' for x in changes]
bars = plt.bar(regions, normalized, color=colors)

# 그래프 꾸미기
plt.title('2015-2025 지역별 인구 변동', fontsize=16)
plt.xlabel('행정구역', fontsize=12)
plt.ylabel('정규화 인구변동량', fontsize=12)
plt.xticks(rotation=90)

# 값 레이블 추가
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}',
             ha='center', va='bottom',
             color='red', fontsize=10)

plt.tight_layout()
plt.savefig('population_change.png')
plt.show()