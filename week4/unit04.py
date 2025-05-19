#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
plt.plot([10, 20, 30, 40])
plt.show()


# In[2]:


import matplotlib.pyplot as plt
plt.plot([1,2,'k',4], [12, 43, 25, 15])
plt.show()


# In[2]:


import matplotlib.pyplot as plt
plt.title('plotting')
plt.plot([10, 20, 30, 40])
plt.show()


# In[5]:


import matplotlib.pyplot as plt
plt.title('legend')
plt.plot([10, 20, 30, 40], label ='오름차순') # 증가를 의미하는 asc 범례 (한글 깨짐)
plt.plot([40, 30, 20, 10], label ='desc') # 감소를 의미하는 desc 범례
plt.legend()
plt.show()


# In[6]:


import matplotlib.pyplot as plt
plt.title('color') # 제목 설정
plt.plot([10, 20, 30, 40], color ='skyblue', label ='skyblue') # 하늘색 그래프
plt.plot([40, 30, 20, 10], 'pink', label ='pink') # 분홍색 그래프
plt.legend() # 범례 표시
plt.show()


# In[3]:


import matplotlib.pyplot as plt
plt.title('linestyle') #제목 설정
plt.plot([10, 20, 30, 40], color ='r', linestyle ='--', label
='dashed') # 빨간색 dashed 그래프
plt.plot([40, 30, 20, 10], color ='g', ls =':', label ='dotted')
# 초록색 dotted 그래프
plt.legend() # 범례 표시
plt.show()


# In[4]:


import matplotlib.pyplot as plt
plt.title('marker') # 제목 설정
plt.plot([10, 20, 30, 40], 'r.', label ='circle') # 빨간색 원형 마커 그래프
plt.plot([40, 30, 20, 10], 'g^', label ='triangle up')
# 초록색 삼각형 마커 그래프
plt.legend() # 범례 표시
plt.show()


# In[ ]:




