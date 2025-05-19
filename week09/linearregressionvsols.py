# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:07:37 2020 @author: Park
"""

# ols()는 통계적 요약(표준오차, R² 등)을 제공하는 장점이 있고, 
# LinearRegression()은 빠르고 단순한 예측에 적합합니다.
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# 데이터 생성
data = {'x' : [156.5, 160.6, 169.5, 167.9, 154.8, 163.0],
        'y' : [51.7, 54.8, 62.3, 61.3, 49.8, 55.8]}
data = pd.DataFrame(data)

# 1. sklearn의 LinearRegression 사용
lr_model = LinearRegression()
lr_model.fit(data[['x']], data['y'])
sklearn_pred = lr_model.predict([[165]])[0]


# 2. statsmodels의 ols 사용
ols_model = smf.ols('y ~ x', data=data).fit()
ols_pred = ols_model.predict(pd.DataFrame({'x': [165]}))[0]

# 결과 출력
print(f"sklearn LinearRegression 예측값: {sklearn_pred:.3f}")
print(f"statsmodels ols 예측값       : {ols_pred:.3f}")

#######################################################  ols 
ols_summary = ols_model.summary()
print(ols_summary)


####################################################### 그래프
import numpy as np
df=data
model = LinearRegression()
model.fit(df[['x']], df['y'])
x_pred = 165
y_pred = model.predict([[x_pred]])[0]
x_vals = np.linspace(df['x'].min()-1, df['x'].max()+1, 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(df['x'], df['y'], color='blue', label='Data Points')
plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.scatter(x_pred, y_pred, color='green', s=100, label=f'Prediction (x=165, y={y_pred:.2f})')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Linear Regression: Height vs. Weight')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()