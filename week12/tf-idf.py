import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '사과',
  '바나나',
  '사과 바나나 쥬스',
  '바나나 바나나 과일',
  '수박'
] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()
N = len(docs) # 총 문서의 수

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t,d)* idf(t)

##################################################################
result = []
for i in range(N): # 각 문서에 대해서 아래 명령을 수행
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]        
        result[-1].append(tf(t, d))
tf_ = pd.DataFrame(result, columns = vocab)
tf_

##################################################################
"""
idf_result= pd.DataFrame(result, columns = vocab) # idf_result=tf_
idf_result.astype(float) 
for i in range(N):
    idf_result.iloc[i]= \
        idf_result.iloc[i].map(lambda x: round(log(N/(1+x)),2))
    #idf_result.iloc[i]=log(4/(idf_result.iloc[i]+1.0))
"""    
##################################################################
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))
idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
idf_

##################################################################
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t,d))
tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_