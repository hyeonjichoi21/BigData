import pandas as pd
data=pd.read_csv("mtcars.csv")
print(data)

dir(pd)
pd.read_csv.__doc__
print(pd.read_csv.__doc__)
print(pd.DataFrame.head.__doc__)

print(data.head())
print(data.shape)
print(type(data))
print(data.columns)
print(data.describe())
print(data['hp'].describe())
print(data['gear'].unique())
print(data['cyl'].unique())

print(data.info())

#print(data.corr())
X=data.drop(columns='mpg') # mpg를 예측하고 싶은 거임 mpg는 연속형 데이터니까 (분류 X, 회귀O) (타겟 변수에 따라 분류가 되기도 ㅋ 회귀가 되기도 ㅋ)
Y=data['mpg']

# 필요없는 열 삭제 - 'Unnamed: 0' 자동차 이름
X=X.iloc[:, 1:]

# 결측치 확인 - 평균값으로 대치
X.isnull()
X.isnull().sum()
X_cyl_mean=X['cyl'].mean()
X['cyl']=X['cyl'].fillna(X_cyl_mean)

X.isnull()
X.isnull().sum()
X_cyl_median=X['qsec'].median()
X['qsec']=X['qsec'].fillna(X_cyl_median)

X=X.dropna()

# 잘못된 값 바꾸기 '*3'->'3'
print(X['gear'].unique())
X['gear']=X['gear'].replace('*3', '3').replace('*5', 5)
print(X.info())

# 'cyl' 최대치 이상값 처리
X_describe=X.describe()
X_iqr=X_describe.loc['75%']-X_describe.loc['25%']
X_iqrmax=X_describe.loc['75%']+(1.5*X_iqr)
X_cyl_list=X.index[X['cyl']>X_iqrmax['cyl']].tolist()
X.loc[X_cyl_list, 'cyl']=X_iqrmax['cyl']

# 'hp' 최소치 이상값 처리
X_describe=X.describe()
#X_iqr=X_describe.loc['75%']-X_describe.loc['25%']
X_hpmin=X_describe.loc['25%']-(1.5*X_iqr)
X_hp_list=X.index[X['hp']<X_hpmin['hp']].tolist()
X.loc[X_hp_list, 'hp']=X_hpmin['hp']

# outlier 찾아보기
def outlier(data, column):
    mean=data[column].mean()
    std=data[column].std()
    lowest=mean-(std*1.5)
    highest=mean+(std*1.5)
    print('최소', lowest, '최대', highest)
    outlier_index=data[column][ (data[column] < lowest) | (data[column] > highest)].index
    return outlier_index

print(outlier(X,'qsec'))
X.loc[24,'qsec']=42.245
#print(X.loc[outlier(X,'qsec'), 'qsec'])
print(outlier(X,'carb'))
X.loc[29,'carb']=5.235
X.loc[30,'carb']=5.235
#print(X.loc[outlier(X,'carb'), 'carb'])

# 데이터 스케일링
import sklearn; 
dir(sklearn)
print(sklearn.__doc__)
sklearn.__all__
import sklearn.preprocessing; dir(sklearn.preprocessing)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
temp=X[['qsec']]
scaler.fit_transform(temp)
qsec_s_scaler=pd.DataFrame(scaler.fit_transform(temp))
print(qsec_s_scaler.describe())
X['qsec']=qsec_s_scaler

# 데이터타입 변경
print(X.info())
X['gear']=X['gear'].astype('object')

# 인코딩 - One-Hot Encoding
pd.get_dummies(X['am'])
pd.get_dummies(X['am'], drop_first=True)
pd.get_dummies(X, drop_first=True)
# 인코딩 - Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder() 
print(encoder.fit_transform(X['am']))
# 인코딩 - replace
X['am'].replace('manual',0).replace('auto', 1) 
X=X.drop(columns='am')

#파생변수 만들기
X['wt'] < 3.3
X.loc[ X['wt'] < 3.3, 'wt_class']=0
X.loc[ X['wt'] >= 3.3, 'wt_class']=1
X['qsec_4']=X['qsec']*4
X.drop(columns='qsec')

# 학습데이터/테스트데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.3, random_state=10)

# 모델학습 Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train, y_train)
y_train_predicted=model.predict(x_train)
y_test_predicted=model.predict(x_test)

dir(LinearRegression)
print(sklearn.linear_model.LinearRegression.__doc__)
print(model.intercept_)
print(model.coef_)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
# RSME
from sklearn.metrics import r2_score
print(r2_score(y_train, y_train_predicted))
print(r2_score(y_test, y_test_predicted))
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_test_predicted))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_test_predicted))



# random forest regressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=10)
model.fit(x_train, y_train)
y_train_predicted=model.predict(x_train)
y_test_predicted=model.predict(x_test)
# RSME
from sklearn.metrics import r2_score
print(r2_score(y_train, y_train_predicted))
print(r2_score(y_test, y_test_predicted))
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_test_predicted))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_test_predicted))




