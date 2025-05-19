# -*- coding: utf-8 -*-
"""
빅데이터 - 3주차 괴제 - 컴퓨터공학전공 20210905 최현지
"""

import pandas as pd

#♣ 질의 [3-1]  emp.csv를 읽어서 DataFrame emp 만들기
emp = pd.read_csv('emp.csv')
print(emp) # 그냥 emp 치는 거랑 똑같음 

#♣ 질의 [3-2] SELECT * FROM Emp;
emp
emp[:]
emp[:][:]

emp.loc[:] # 레이블(행, 열 ★이름★ 으로 출력, 끝값 포함)
emp.loc[:,:]

emp.iloc[:] # 정수인덱스 (행, 열 ★위치★로 출력, 끝값 미포함)
emp.iloc[:,:]

#♣ 질의 [3-3] SELECT ename FROM Emp;
emp.ENAME
emp['ENAME']
emp.loc[:, 'ENAME']
emp.loc[:]['ENAME']
emp.iloc[:,1] #ENAME이 첫 번째 인덱스 열에 해당하기 때문 ~ !

#♣ 질의 [3-4] SELECT ename, sal FROM Emp;
emp[['ENAME', 'SAL']]
print(emp[['ENAME', 'SAL']])

emp.loc[:, ['ENAME', 'SAL']]
emp.loc[0:13, ['ENAME', 'SAL']]

emp.iloc[:, [1,5]]
emp.iloc[0:14, [1,5]] # iloc는 ★끝을 포함하지 않기★ 때문에 0:13으로 하면 안됨 주의!


#♣ 질의 [3-5] SELECT DISTINCT job FROM Emp;

# emp.JOB - 일반 출력
# emp['JOB'] - 일반 출력

emp['JOB'].unique() # 결과는 Array
emp['JOB'].drop_duplicates() # 결과는 Series 


#♣ 질의 [3-6] SELECT * FROM Emp WHERE sal < 2000;
# SAL이 2000보다 작은 행 필터링
emp[emp['SAL'] < 2000]


#♣ 질의 [3-7] SELECT * FROM Emp WHERE sal BETWEEN 1000 AND 2000;
emp[(emp['SAL']<=2000) & (emp['SAL']>=1000)] # Python - Pandas 는 &&이 아니라 &다 !!

#♣ 질의 [3-8] SELECT * FROM Emp WHERE sal >= 1500 AND job= ‘SALESMAN’;
emp[(emp.SAL >= 1500) & (emp.JOB=='SALESMAN')] # 괄호로 감싸주고 &만 되고 (AND 안됨), = == 차이 주의 

#♣ 질의 [3-9] SELECT * FROM Emp WHERE job IN ('MANAGER', 'CLERK');
# 방법1 <- 내가 풀음
emp[(emp.JOB == 'MANAGER') | (emp.JOB == 'CLERK')]

# 방법2 <- perplexity (feat.pro-ver)이 풀음
emp[emp.JOB.isin(['MANAGER', 'CLERK'])] # ([ ])<- 대괄호 감싸는 거 주의 !!


#♣ 질의 [3-10] SELECT * FROM Emp WHERE job NOT IN ('MANAGER', 'CLERK');
# Python - Pandas에서의 ★부정★ : ~ 
emp[~((emp.JOB == 'MANAGER') | (emp.JOB == 'CLERK'))]

emp[~emp.JOB.isin(['MANAGER', 'CLERK'])]


#♣ 질의 [3-11] SELECT ename, job FROM Emp WHERE ename LIKE 'BLAKE';

# emp[emp['ENAME'] == 'BLAKE', ['ENAME','JOB'] ] <- 이상하게 나옴.. 
emp.loc[emp['ENAME'] == 'BLAKE', ['ENAME', 'JOB']]


#♣ 질의 [3-12] SELECT ename, job FROM Emp WHERE ename LIKE '%AR%';
emp.loc[emp['ENAME'].str.contains('AR'),  ['ENAME', 'JOB']]
emp.loc[emp['ENAME'].str.contains('AR', na=False), ['ENAME', 'JOB']] # nan값 제거 



#♣ 질의 [3-13] SELECT * FROM Emp WHERE ename LIKE '%AR%' AND sal >= 2000;
emp[(emp.SAL >= 2000) & (emp.ENAME.str.contains('AR'))]  # 괄호 필수 !! 그래야 제대로 인식됨

#♣ 질의 [3-14] SELECT * FROM Emp ORDER BY ename;
emp.sort_values(by='ENAME')


#♣ 질의 [3-15] SELECT SUM(sal) FROM Emp;
emp.SAL.sum() # 대문자로 표기, 소문자는 호환 안 됨 
emp['SAL'].sum()


#♣ 질의 [3-16] SELECT SUM(sal) FROM Emp WHERE job LIKE 'SALESMAN'; 
emp.loc[emp['JOB'] == 'SALESMAN', 'SAL'].sum() # loc쓸 거면 '속성'으로 정확하게 명시

# perplexity 풀이 
emp[emp['JOB'] == 'SALESMAN']['SAL'].sum()


#♣ 질의 [3-17] SELECT SUM(sal), AVG(sal), MIN(sal), MAX(sal) FROM Emp; 
# emp.SAL.sum().mean().min().max() <- 이렇게는 안 됨

# perplexity 의 도움을 받음 ...
sal_Sum = emp.SAL.sum()
sal_Avg = emp.SAL.mean() # 파이썬 - 판다스 에서는 평균함수 ★mean()★
sal_Min = emp.SAL.min()
sal_Max = emp.SAL.max()

# 결과 전체 출력 
print(f"SUM: {sal_Sum} , AVG: {sal_Avg}, MIN: {sal_Min}, MAX: {sal_Max}")


#♣ 질의 [3-18] SELECT COUNT(*) FROM Emp; 
emp.shape[0]


emp.value_counts() # 이건 고유한 값이 각각 몇 번 나타나는지를 계산하여 반환하는 함수


#♣ 질의 [3-19] SELECT COUNT(*), SUM(sal) FROM Emp GROUP BY job;

# 방법1 - 집계함수 "별도" 계산
# JOB별로 SAL의 COUNT(*) 계산
emp.groupby('JOB')['SAL'].count()

# JOB별로 SAL의 SUM 계산
emp.groupby('JOB')['SAL'].sum()

""""""""""""""""""""""""""""""""""""""""""""""""

# 방법2 - 집계함수 "한꺼번에" 계산 
emp.groupby('JOB').agg(
    COUNT=('SAL', 'count'),  # SAL 열에서 값의 개수를 계산 (COUNT)
    SUM=('SAL', 'sum')       # SAL 열의 합계를 계산 (SUM)
)

emp.groupby('JOB').agg(
    COUNT=('SAL','count'), # agg 함수에서는 (열 이름, 출력될 열 이름 ) 형식이다. 
    SUM=('SAL', 'sum')    
    
)

#♣ 질의 [3-20] SELECT * FROM Emp WHERE comm IS NOT NULL;
emp[emp['COMM'].notnull()]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#♣ [질의4-0] emp.csv를읽어서DataFrame emp 만들기
import pandas as pd 
emp = pd.read_csv('emp.csv')

emp

#♣ [질의4-1] emp에 age열을 만들어 다음을 입력하여라(14명) 
# [30,40,50,30,40,50,30,40,50,30,40,50,30,40]
#ages = [30, 40, 50,30,40,50,30,40,50,30,40,50,30,40]
emp['age'] = [30, 40, 50,30,40,50,30,40,50,30,40,50,30,40]



#♣ [질의4-2] INSERT INTO Emp(empno, ename, job) Values (9999,  ‘ALLEN’, ‘SALESMAN’)
new_row = pd.DataFrame({'EMPNO': [9999], 'ENAME': ['ALIEN'], 'JOB': ['SALESMAN']})


# 기존 DataFrame과 새로운 행을 concat으로 결합
emp = pd.concat([emp, new_row], ignore_index=True)

emp


#♣ [질의4-3] emp의 ename=‘ALLEN’ 행을 삭제하여라
# (DELETE FROM emp WHERE ename LIKE ‘ALLEN’;)
emp = emp[emp['ENAME'] != 'ALIEN']
emp


#♣ [질의4-4] emp의 hiredate 열을 삭제하여라
# (ALTER TABLE emp DROP COLUMN hiredate;)
emp= emp.drop(columns = ['HIREDATE'], errors = 'ignore')
emp.head()


#♣ [질의4-5] emp의 ename=‘SCOTT’의 sal을 3000으로 변경하여라
# (UPDATE emp SET sal=3000 WHERE ename LIKE ‘SCOTT’;

emp.loc[emp['ENAME'] == 'SCOTT', 'SAL'] = 3000
emp


#♣ [질의5-1] emp의 sal 컬럼을 oldsal 이름으로변경하여라. 
# (ALTER TABLE emp RENAME sal TO oldsal;)
emp.rename(columns={'SAL':'OLDSAL'}, inplace=True)
emp.head()


#♣ [질의5-2] emp에 newsal 컬럼을 추가하여라, 값은 oldsal 컬럼값
# (ALTER TABLE emp ADD newsal …;)
emp['NEWSAL'] = emp['OLDSAL']
emp.head()


#♣ [질의5-3] emp의 oldsal 컬럼을 삭제하여라
# (ALTER TABLE emp DROP COLUMN oldsal;)
emp = emp.drop(columns = ['OLDSAL'], errors='ignore')
emp.head()
























