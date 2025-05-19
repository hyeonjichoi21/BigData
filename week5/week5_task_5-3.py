import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 읽기 (UTF-8 인코딩)
df = pd.read_csv('age.csv', encoding='cp949')

# 2. 사용자 입력 받기
target_region = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요: ')

# 3. 대상 지역 데이터 추출
target_row = df[df['행정구역'].str.contains(target_region)].iloc[0]
age_columns = df.columns[3:]  # 0세~ 열 선택
total_population = target_row['총인구수']
target_ratios = target_row[age_columns].astype(int) / total_population

# 4. 모든 지역에 대한 유사도 계산
def calculate_similarity(row):
    ratios = row[age_columns].astype(int) / row['총인구수']
    return ((ratios - target_ratios) ** 2).sum()

# 유사도 계산 및 정렬
similarity_df = df.copy()
similarity_df['similarity'] = df.apply(calculate_similarity, axis=1)
sorted_df = similarity_df.sort_values(by='similarity')

# 5. 시각화
plt.figure(figsize=(12, 6), dpi=300)
plt.rc('font', family='Malgun Gothic')
plt.title(f'{target_region}과 가장 유사한 인구 구조를 가진 지역')

# 대상 지역 플롯
plt.plot(target_ratios.values, label=target_region, linewidth=3)

# 상위 5개 지역 플롯
for i in range(1, 6):  # 0번은 자기 자신이므로 1~5번 선택
    region = sorted_df.iloc[i]
    ratios = region[age_columns].astype(int) / region['총인구수']
    plt.plot(ratios.values, label=region['행정구역'], alpha=0.7)
    
 # X축 설정 (0,20,40,60,80,100 세 단위로 표시)
x_ticks = range(0, len(age_columns), 20)  # 20년 간격 위치
x_labels = [f'{age}세' for age in range(0, 101, 20)]  # 0세~100세
plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)

plt.xlabel('연령대')
plt.ylabel('인구 비율')
plt.legend()
plt.tight_layout()
plt.show()