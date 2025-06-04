import pandas as pd
data=pd.read_csv("mtcars.csv")
print(data)

dir(pd)
pd.read_csv.__doc__
print(pd.read_csv.__doc__)
print(pd.DataFrame.head.__doc__)

dir(pd.DataFrame) ###

print(data.head())
print(data.shape)
print(type(data))
print(data.columns)

print(type(data.columns))

print(data.describe()) # 평균 이런값들.. 수치 데이터만 
print(data['hp'].describe()) 
print(data['gear'].unique()) # 값이 몇개인지
print(data['cyl'].unique()) # 값이 몇개인지

print(data.info())

print(data.corr())
X=data.drop(columns='mpg')
Y=data['mpg']
