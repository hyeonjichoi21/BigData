import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv('playtennis.csv')
print("원본 데이터:")
print(df)
print(f"\n데이터 크기: {df.shape}")

# 2. 데이터 전처리 (LabelEncoder 사용)
le = LabelEncoder()
df_encoded = df.copy()

# 모든 컬럼에 대해 Label Encoding 적용
for col in df.columns:
    df_encoded[col] = le.fit_transform(df[col])

print(f"\n인코딩된 데이터:")
print(df_encoded)

# 3. X, y 분리
X = df_encoded.drop('PlayTennis', axis=1)  # 독립변수
y = df_encoded['PlayTennis']               # 종속변수

# 4. 학습용/테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n훈련 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")

# 5. 의사결정 트리 모델 학습
dt_clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_clf.fit(X_train, y_train)

# 6. 테스트셋 예측
y_pred = dt_clf.predict(X_test)

# 7. 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== 성능 평가 결과 (테스트 데이터 기준) ===")
print(f"Accuracy: {accuracy:.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. 의사결정 트리 시각화
plt.figure(figsize=(20, 10))
plot_tree(dt_clf,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True)
plt.title("Decision Tree for Play Tennis Dataset (Train/Test Split)")
plt.show()
