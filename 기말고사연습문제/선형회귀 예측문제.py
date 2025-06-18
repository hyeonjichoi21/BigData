from sklearn.linear_model import LinearRegression

# 1. 학습 데이터 준비
X = [[1], [2], [3]]
y = [2, 4, 6]

# 2. 모델 학습
model = LinearRegression()
model.fit(X, y)

# 3. 기울기와 절편 확인
print("기울기 (coef_) :", model.coef_[0])
print("절편 (intercept_) :", model.intercept_)

# 4. X=10일 때 예측
x_input = [[10]]
y_pred = model.predict(x_input)
print("X=10일 때 예측값 :", y_pred[0])



"""
+ 예상 문제
1. 회귀 계수 보고 식 작성하기

cylinders: -0.14, displacement: 0.01, weight: -0.01, acceleration: 0.20, model_year: 0.76, 절편: -17.55

👉 회귀식을 작성하시오

Y = -0.14 × cylinders + 0.01 × displacement - 0.01 × weight + 0.20 × acceleration + 0.76 × model_year - 17.55

-------------------------------------------------------------------------------

2. 예측값 계산하기

회귀식: Y = 0.5 × X1 + 0.3 × X2 + 2.0  
X1=4, X2=6일 때 Y는?

정답: Y = 0.5×4 + 0.3×6 + 2.0 = 2.0 + 1.8 + 2.0 = ""5.8""

""" 
