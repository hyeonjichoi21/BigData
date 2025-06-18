# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 19:18:00 2025

@author: chj
"""
import matplotlib.pyplot as plt
"""step1 라인플롯(선 그래프) 차트 그리기"""

#1. 데이터 준비
x = [2016, 2017, 2018, 2019, 2020]
y = [350, 410, 520, 695, 543]

#2. x축과 y축 데이터를 지정하여 라인플롯 생성
plt.plot(x,y)

#3. 차트 제목 설정
plt.title('Annual sales')

#4. x축 레이블(이름) 설정
plt.xlabel('years')
plt.ylabel('sales')

#5. 라인플롯 표시
plt.show()

"""step2 바차트(막대 그래프) 차트 그리기"""

#1. 데이터 준비
y1 = [350, 410, 520, 695]
y2 = [200, 250, 385, 350]
