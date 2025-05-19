# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:33:39 2025

@author: gusdk
"""

import numpy as np
import pandas as pd

# 3주차 과제 질의 1)
emp = pd.read_csv('emp.csv')
emp # print(emp)랑 똑같음

# 질의 2) SELECT * FROM Emp
emp
emp[:]
emp[:][:]

emp.loc[:]
emp.loc[:,:]

emp.iloc[:]
emp.iloc[:,:]


# 질의 3) [질의 3-3] SELECT ename FROM Emp;
emp.ENAME
emp['ENAME']
emp.loc[:, 'ENAME']
emp.loc[:]['ENAME']
emp.iloc[:,1]


# [질의 3-4] SELECT ename, sal FROM Emp;
emp[['ENAME','SAL']] # dataFrame
emp.loc[:, ['ENAME', 'SAL']]
emp.iloc[:, [1,5]]

emp.loc[0:13, ['ENAME', 'SAL']] # loc은 끝을 포함
emp.iloc[0:13, [1,5]] # iloc은 끝을 포함 X
emp.iloc[0:14, [1,5]]



# [질의 3-5] SELECT DISTINCT job FROM Emp;
emp['JOB']
emp['JOB'].unique() # 결과는 Array
emp['JOB'].drop_duplicates() # 결과는 Series



# [질의 3-6] SELECT * FROM Emp WHERE sal < 2000;



[질의 3-7] SELECT * FROM Emp WHERE sal BETWEEN 1000 AND 2000;
# [질의 3-8] SELECT * FROM Emp WHERE sal >= 1500 AND job= ‘SALESMAN’;
[질의 3-9] SELECT * FROM Emp WHERE job IN ('MANAGER', 'CLERK');
[질의 3-10] SELECT * FROM Emp WHERE job NOT IN ('MANAGER', 'CLERK');
[질의 3-11] SELECT ename, job FROM Emp WHERE ename LIKE 'BLAKE';
[질의 3-12] SELECT ename, job FROM Emp WHERE ename LIKE '%AR%';
[질의 3-13] SELECT * FROM Emp WHERE ename LIKE '%AR%' AND sal >= 2000;










































