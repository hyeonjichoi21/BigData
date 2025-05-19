#############################################################################
##### 프로젝트 1. [기술통계분석 + 그래프] - 와인품질등급 예측하기  #############
##################################################################

#1. 엑셀에서 열 구분자를 세미콜론으로 인식시키기 --------------------------
import pandas as pd

red_df = pd.read_csv('DATA/winequality-red.csv', sep = ';', header = 0, engine = 'python')

white_df = pd.read_csv('DATA/winequality-white.csv', sep = ';', header = 0, engine= 'python')

red_df.to_csv('DATA/winequality-red2.csv', index = False)

white_df.to_csv('DATA/winequality-white2.csv', index = False)

# 4.2 데이터 병합하기 ------------------------------------------------------------
# 1. 레드 와인과 화이트 와인 파일 합치기
red_df.head()

red_df.insert(0, column = 'type', value = 'red')

red_df.head()

red_df.shape

white_df.head()

white_df.insert(0, column = 'type', value = 'white')

white_df.head()

white_df.shape

wine = pd.concat([red_df, white_df])

wine.shape

wine.to_csv('DATA/wine.csv', index = False)


#5 데이터 탐색 -----------------------------------------

#1. 기본 정보 확인하기
print(wine.info())

#2. 함수를 사용해 기술 통계 구하기
wine.columns = wine.columns.str.replace(' ', '_') #
wine.head()
wine.describe()
sorted(wine.quality.unique())
wine.quality.value_counts()


# 6 데이터 모델링 -----------------------------------

# 6.1 describe( ) 함수로 그룹 비교하기
wine.groupby('type')['quality'].describe()
wine.groupby('type')['quality'].mean()
wine.groupby('type')['quality'].std()
wine.groupby('type')['quality'].agg(['mean', 'std'])

# 6.2 t-검정과 회귀 분석으로 그룹 비교하기
### cmd창에서 통계패키지 설치하기:  pip install statsmodels 
from scipy import stats
from statsmodels.formula.api import ols, glm
red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']
stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = False)

Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'

regression_result = ols(Rformula, data = wine).fit()

regression_result.summary()

# 6.3 회귀 분석 모델로 새로운 샘플의 품질 등급 예측하기
sample1 = wine[wine.columns.difference(['quality', 'type'])]
sample1 = sample1[0:5][:]
sample1_predict = regression_result.predict(sample1)
sample1_predict
wine[0:5]['quality']

data = {"fixed_acidity" : [8.5, 8.1], "volatile_acidity":[0.8, 0.5],
"citric_acid":[0.3, 0.4], "residual_sugar":[6.1, 5.8], "chlorides":[0.055,
0.04], "free_sulfur_dioxide":[30.0, 31.0], "total_sulfur_dioxide":[98.0,
99], "density":[0.996, 0.91], "pH":[3.25, 3.01], "sulphates":[0.4, 0.35],
"alcohol":[9.0, 0.88]}

sample2 = pd.DataFrame(data, columns= sample1.columns)
sample2


sample2_predict = regression_result.predict(sample2)
sample2_predict


# 7 결과 시각화 --------------------------------------------
#7.1 와인 유형에 따른 품질 등급 히스토그램 그리기
###cmd 창에서 패키지 설치하기:   pip install seaborn 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')

sns.histplot(red_wine_quality, stat='density', kde = True, color = "red", label = 'red wine')
sns.histplot(white_wine_quality, stat='density', kde = True, label = 'white wine')
plt.title("Quality of Wine Type")
plt.legend()
plt.show()


#7.2 부분 회귀 플롯으로 시각화하기
import statsmodels.api as sm
others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))
p, resids = sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)
plt.show()

fig = plt.figure(figsize = (8, 13))
sm.graphics.plot_partregress_grid(regression_result, fig = fig)
plt.show()


#############################################################################
##### 프로젝트 2. [상관 분석 + 히트맵] - 타이타닉호 생존율 분석하기  #############
##################################################################

#3 데이터 수집
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")
titanic.to_csv('DATA/titanic.csv', index = False)

#4 데이터 준비
titanic.isnull().sum()
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'].value_counts()
titanic['embarked'] = titanic['embarked'].fillna('S')
titanic['embark_town'].value_counts()
titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
titanic['deck'].value_counts()
titanic['deck'] = titanic['deck'].fillna('C')
titanic.isnull().sum()

#5 데이터 탐색
#5.1 데이터의 기본 정보 탐색하기
titanic.info()
titanic.survived.value_counts()

#5.2 차트를 그려 데이터를 시각적으로 탐색하기
import matplotlib.pyplot as plt
f,ax = plt.subplots(1, 2, figsize = (10, 5))
titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')
plt.show()

#5.3 등급별 생존자 수를 차트로 나타내기
sns.countplot(x = 'pclass', hue = 'survived', data = titanic)
plt.title('Pclass vs Survived')
plt.show()


#6 데이터 모델링
#6.1 상관 분석을 위한 상관 계수 구하고 저장하기
titanic2 = titanic.select_dtypes(include=[int, float,bool])
titanic2.shape
titanic_corr = titanic2.corr(method = 'pearson')
titanic_corr
titanic_corr.to_csv('DATA/titanic_corr.csv', index = False)

#6.3 특정 변수 사이의 상관 계수 구하기
titanic['survived'].corr(titanic['adult_male'])
titanic['survived'].corr(titanic['fare'])


#7 결과 시각화
#7.1 산점도로 상관 분석 시각화하기
sns.pairplot(titanic, hue = 'survived')
plt.show()

#7.2 두 변수의 상관관계 시각화하기
sns.catplot(x = 'pclass', y = 'survived', hue = 'sex', data = titanic, kind = 'point')
plt.show()


#7.3 변수 사이의 상관 계수를 히트맵으로 시각화하기
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7


titanic['age2'] = titanic['age'].apply(category_age)
titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1
titanic.to_csv('DATA/titanic3.csv', index = False)
heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]
colormap = plt.cm.RdBu
sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True, annot_kws = {"size": 10})

plt.show()











