# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:56:03 2025

2개의 리스트를 읽어서 공통된 데이터가 있으면 True, 아니면 None을 반환하는 프로그램
을 작성하여라

'S
"""

"""
def common_data(list1, list2):
    a = 0; #default값 설정
    for i in list1:
        for j in list2:
            if (i == j ):
                a = 1;
                

if (a==1):
    print("True")
else:
    print("False")
"""           

# 교수님 풀이
def common_data(list1, list2):
    result = false
    for x in list1:
        if x in list2:
            result = True 
    return result

print(common_data([1,2,3,4,5], [5,6,7,8,9])) # => True
print(common_data([1,2,3,4,5], [6,7,8,9])) # => False



"""
문자열을 읽어서 리스트에 원소들이 있으면 True, 아니면 False를 반환하는 프로그램을
작성하여라
"""
# 교수님 풀이 <- 인데 다 못품. 좀 더 생각해봐라. 
"""
def test(lst, str1):
    result = False
    str2 = str1.split('./') 
    print(str2)
    for x in lst:
        if x in str2:
            result = True
    return result 
"""


def test(lst, str1):
    result = [el for el in lst if(el in str1)]
    return bool(result)


"""
# 함수 정의 
def test(lst, str1):
    # 리스트 lst의 각 원소가 str1에 포함되어 있는지 확인
    return any(sub in str1 for sub in lst)
"""

str1 = "https://www.w3resource.com/python-exercises/list/"
lst = ['.com', '.edu', '.tv']
print(test(lst,str1)) #True


str1 = "https://www.w3resource.net"
lst = ['.com', '.edu', '.tv']
print(test(lst,str1)) #False
