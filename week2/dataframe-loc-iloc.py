import numpy as np
import pandas as pd

# creating
df = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])
df['1반']['중간고사']
type(df)

# indexing-selection-assigning
df['중간고사']
type(df['중간고사'])
df['중간고사'][0:2]
type(df['중간고사'][0])
type(df['중간고사'][0:2])
df['중간고사']['1반':'2반']
type(df['중간고사']['1반':'2반'])

# loc
df.loc['1반']
type(df.loc['1반'])
df.loc[:, '중간고사']
type(df.loc[:, '중간고사'])
df.loc['1반':'2반']['중간고사']
df.loc['1반', '중간고사']
type(df.loc['1반', '중간고사'])
df.loc['1반'][0]

df.iloc[0]
type(df.iloc[0])
df.iloc[0]['중간고사']
type(df.iloc[0]['중간고사'])

df.loc[df.성적 == 'B']
df.loc[(df.성적 == 'A') & (df.중간고사 >= 90)]
df.loc[df.성적.isin(['B', 'C'])]

## summary function and maps
df.describe()
df.중간고사.describe()
df.head(1)
df.중간고사.unique()
df.중간고사.mean()
df.중간고사.value_counts()
df_mean = df.중간고사.mean()
df.중간고사.map(lambda p: p - df_mean)

## grouping and sorting
df.groupby('중간고사').중간고사.count()
df.groupby('중간고사').중간고사.min()
df.groupby(['중간고사']).중간고사.agg([len, min, max])
df.sort_values(by='중간고사')
df.sort_values(by='중간고사', ascending=False)
df.sort_index(ascending=False)

# data types and missing values
df.dtypes
df.중간고사.dtypes
df.loc['6반']=[10, 10, np.nan]
df[pd.isnull(df.성적)]

# renaming and combining
df.rename(columns={'성적': '등급'})
df.rename_axis("반이름", axis='rows')

df1 = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])

df0=pd.concat([df, df1])

