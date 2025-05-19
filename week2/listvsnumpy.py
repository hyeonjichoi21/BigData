# list

myarray = [1, 2, 3]
print (type(myarray))           # 출력 "<type> 
#print (mynumpy.shape)      # 
print (myarray[0], myarray[1], myarray[2]) # 출력 "1 2 3"
myarray[0] = 5                  # 요소를 변경
print (myarray)                  # 출력 "[5, 2, 3]"
print (sum(myarray))
print (myarray.sort())    	      # sort
print (myarray*2)

yourarray = [[1,2,3],[4,5,6]]   # 원소 2개인 배열 생성
#print (yournumpy.shape)   # 
print (yourarray[0])
print (yourarray[0][0], yourarray[0][1], yourarray[1][0])  

x = [[1,2],[3,4]]
y = [[5,6],[7,8]]
print (x + y)
#print (add(x, y))


# numpy
import numpy as np
mynumpy = np.array([1, 2, 3])  # rank가 1인 배열 생성
print (type(mynumpy))            # 출력 "<type> 
print (mynumpy.shape)           # 출력 "(3,)" 일차원배열이면서, 길이가 3이라는 뜻
print (mynumpy[0], mynumpy[1], mynumpy[2])   # 출력
mynumpy[0] = 5                    # 요소를 변경
print (mynumpy)                    # 출력 "[5, 2, 3]"
print (np.sum(mynumpy))
print (np.sort(mynumpy))         # sort
print (mynumpy*2)               # 리스트 곱하기와 큰 차이!!

yournumpy = np.array([[1,2,3],[4,5,6]])   # rank가 2인 배열 생성
print (yournumpy.shape)          # 출력 "(2, 3)“
print (yourarray[0])
print (yournumpy[0, 0], yournumpy[0, 1], yournumpy[1, 0])

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]], dtype=np.float64)
print (x + y) # 정수 + 실수 = 실수?
print (np.add(x, y))
