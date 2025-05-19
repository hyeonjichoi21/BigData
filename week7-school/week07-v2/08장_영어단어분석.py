#!/usr/bin/env python
# coding: utf-8

# # 8장. 텍스트빈도분석 - 1.영어단어분석

# ### 영어 단어 분석에 필요한 패키지 준비

# In[ ]:
#get_ipython().system('pip install matplotlib  #최초 한번만 설치:Anaconda에 설치됨')
# In[ ]:
#get_ipython().system('pip install wordcloud  #최초 한번만 설치:Anaconda에 설치됨')
# In[1]:
import pandas as pd
import glob
import re
from functools import reduce

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# # 1. 데이터 준비
# ### 1-1. 파일 병합
# ### - ☺데이터를 다운 받은 시점에 따라 검색결과가 달라지므로, 책에 있는 결과 화면과 다를수 있습니다.☺ -
# In[2]:
all_files = glob.glob('8장_data/myCabinetExcelData*.xls')
all_files #출력하여 내용 확인

# In[3]:
# pip install xlrd <= 설치 필요...
all_files_data = [] #저장할 리스트 
for file in all_files:
    data_frame = pd.read_excel(file)
    all_files_data.append(data_frame)
all_files_data[0] #출력하여 내용 확인

# In[4]:
all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)
all_files_data_concat #출력하여 내용 확인

# In[5]:
all_files_data_concat.to_csv('8장_data/riss_bigdata.csv', encoding='utf-8', index = False)

# ### 1-2. 데이터 전처리 (Pre-processing)
# In[6]:
# 제목 추출
all_title = all_files_data_concat['제목']
all_title #출력하여 내용 확인

# In[7]:
# import nltk
# nltk.download('punkt')    # 해보고 안되면 다음 명령어 실행
# nltk.download('all')      # 5분 이상 소요됨
stopWords = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

# In[8]:
# 아래 2문장이 필요함 import nltk, nltk.download('omw-1.4')
import nltk
nltk.download('omw-1.4')
words = []  
for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title))    
    EnWordsToken = word_tokenize(EnWords.lower())
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords]
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]
    words.append(EnWordsTokenStopLemma)

# In[9]:
print(words)  #출력하여 내용 확인

# In[10]:
words2 = list(reduce(lambda x, y: x+y,words))
print(words2)  #작업 내용 확인

# # 2. 데이터 탐색
# ## 2-1. 단어 빈도 탐색
# In[11]:
count = Counter(words2)
count   #출력하여 내용 확인

# In[12]:
word_count = dict()
for tag, counts in count.most_common(50):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

# #### 여기서 잠깐!! : 검색어로 사용한 big'과 'data' 빈도가 압도적으로 많으므로, 이를 제거한다.

# In[13]:
#검색어로 사용한 'big'과 'data' 항목 제거 하기
del word_count['big']
del word_count['data']

# ## 2-2 단어 빈도 히스토그램
# In[14]:
# 히스토그램 표시 옵션 
plt.figure(figsize=(12,5))
plt.xlabel("word")
plt.ylabel("count")
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='vertical')

plt.show()

# # 3. 분석 모델 구축 및 결과 시각화
# ## 3-1. 연도별 데이터 수

# In[15]:
all_files_data_concat['doc_count'] = 0
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()
summary_year  #출력하여 내용 확인

# In[16]:
plt.figure(figsize=(12,5))
plt.xlabel("year")
plt.ylabel("doc-count")
plt.grid(True)

plt.plot(range(len(summary_year)), summary_year['doc_count'])
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])

plt.show()

# ## 3-2. 워드클라우드
# In[17]:
stopwords=set(STOPWORDS)
wc=WordCloud(background_color='ivory', stopwords=stopwords, width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()
# #### - 워드 클라우드에 나타나는 단어의 위치는 실행 할 때마다 달라진다. ☺

# In[18]:
cloud.to_file("8장_data/riss_bigdata_wordCloud.jpg")

