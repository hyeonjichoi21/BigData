from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
import pandas as pd
import numpy as np

tennis_data = pd.read_csv('playtennis.csv')
tennis_data
df=tennis_data

# 2. 입력(X), 타겟(y) 분리
X = df.drop(columns=['PlayTennis'])
y = df['PlayTennis']

# 3. One-Hot Encoding
X_encoded = pd.get_dummies(X) # 서열형이 아니라, """명목형"""으로 바꾼다 ..? <- 이 방식이 더 정확하다. 어마어마하게 변수 수가 늘어남 

# 4. 결정 트리 모델 학습
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_encoded, y)

# 5. 트리 시각화
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X_encoded.columns, class_names=model.classes_, filled=True)
plt.tight_layout()
plt.show()

