#!/usr/bin/env python
# coding: utf-8

# # 8장. 텍스트빈도분석 - 2) 한글 단어 분석 

# ## 한글 단어 분석을 위한 패키지 준비

# In[1]:


import json
import re

from konlpy.tag import Okt

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud


# # 1. 데이터 준비

# ### 1-1. 파일 읽기

# In[2]:


inputFileName = '8장_data/etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명'
data = json.loads(open(inputFileName+'.json', 'r', encoding='utf-8').read())
data #출력하여 내용 확인


# ### 1-2. 분석할 데이터 추출

# In[3]:


message = ''

for item in data:
    if 'message' in item.keys(): 
        message = message + re.sub(r'[^\w]', ' ', item['message']) +''
        
message #출력하여 내용 확인


# ### 1-3. 품사 태깅 : 명사 추출

# In[4]:


nlp = Okt()
message_N = nlp.nouns(message)
message_N   #출력하여 내용 확인


# ## 2. 데이터 탐색

# ### 2-1. 단어 빈도 탐색

# In[5]:


count = Counter(message_N)

count   #출력하여 내용 확인


# In[6]:


word_count = dict()

for tag, counts in count.most_common(80):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))


# ### 히스토그램

# In[7]:


font_path = "c:/Windows/fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family=font_name)


# In[8]:


plt.figure(figsize=(12,5))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='vertical')

plt.show()


# ### 워드클라우드

# In[9]:


wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[10]:


cloud.to_file(inputFileName + '_cloud.jpg')


# In[ ]:




