# input
import pandas as pd
x_train=pd.read_csv('x_train.csv', encoding='euckr')
x_test=pd.read_csv('x_test.csv', encoding='euckr')
y_train=pd.read_csv('y_train.csv', encoding='euckr')

x_train.head()
y_train.head()
x_test.head()

x_train.describe()
x_train.info()

dir(x_train)
help(x_train.shape)

# 테스트 데이터의 cust_id 저장
x_test_cust_id=x_test['cust_id']
# cust_id 컬럼 삭제
x_train.drop(columns=['cust_id'], inplace=True)
x_test.drop(columns=['cust_id'], inplace=True)
y_train.drop(columns=['cust_id'], inplace=True)

# 결측치 처리
x_train.count()-x_train.notnull().count()
x_test.count()-x_test.notnull().count()
x_train['환불금액']=x_train['환불금액'].fillna(0)
x_test['환불금액']=x_test['환불금액'].fillna(0)

# 라벨인코딩 수행하기
x_train['주구매상품'].unique().size
x_train['주구매지점'].unique().size
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
x_train['주구매상품']=encoder.fit_transform(x_train['주구매상품'])
x_train['주구매지점']=encoder.fit_transform(x_train['주구매지점'])
x_test['주구매상품']=encoder.fit_transform(x_test['주구매상품'])
x_test['주구매지점']=encoder.fit_transform(x_test['주구매지점'])
print(encoder.classes_)

# 파생변수 만들기
condition=x_train['환불금액'] > 0
x_train.loc[condition, '환불금액_new']=1
x_train.loc[~condition, '환불금액_new']=0
x_train.drop(columns=['환불금액'], inplace=True)
condition=x_test['환불금액'] > 0
x_test.loc[condition, '환불금액_new']=1
x_test.loc[~condition, '환불금액_new']=0
x_test.drop(columns=['환불금액'], inplace=True)

# 데이터 스케일링, 표준화 크기로 변환하기
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x_train)
x_train=pd.DataFrame(x, columns=x_train.columns)
x=scaler.transform(x_test)
x_test=pd.DataFrame(x, columns=x_test.columns)

# 불필요한 컬럼 삭제
x_train[['총구매액','최대구매액','환불금액_new']].corr()
x_train.drop(columns=['최대구매액'], inplace=True)
x_test.drop(columns=['최대구매액'], inplace=True)

# 분석
import sklearn.tree
dir(sklearn.tree)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=10, criterion='entropy')
print(model.__doc__)
model.fit(x_train, y_train)
y_test_prob=model.predict_proba(x_test)
result=pd.DataFrame(y_test_prob)[1]
final=pd.concat([x_test_cust_id, result], axis=1).rename(columns={1:'gender'})
final.to_csv('result.csv', index=False)


# train-test split - 모델평가
from sklearn.model_selection import train_test_split
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST=train_test_split(x_train, y_train, test_size=0.2, random_state=10)
print(X_TRAIN.shape)
import sklearn.tree
dir(sklearn.tree)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=10, criterion='entropy')
print(model.__doc__)
model.fit(X_TRAIN, Y_TRAIN)
y_test_prob=model.predict_proba(X_TEST)
Y_TEST_PREDICTED=model.predict(X_TEST)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_TEST, Y_TEST_PREDICTED))








