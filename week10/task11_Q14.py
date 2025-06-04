from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 준비
iris = load_iris()
X = iris.data
y = iris.target

# train/test 분할 (20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# n_estimators 값에 따른 정확도 측정
accuracies = []
n_list = list(range(1, 21))  # 1~20개 트리

for n in n_list:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

for n, acc in zip(n_list, accuracies):
    print(f"n_estimators={n}: 정확도={acc:.3f}")

from sklearn.tree import export_graphviz
import graphviz

# 예시: 5번째 트리 시각화
estimator = clf.estimators_[5]
dot_data = export_graphviz(
    estimator,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True, special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("iris_tree")  # iris_tree.pdf 또는 iris_tree.png로 저장
graph  # 주피터 노트북에서는 바로 출력
