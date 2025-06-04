# # 13장. 텍스트마이닝_감성분석과 토픽분석
# ### - pandas 버전이 1.1.4 이상인지 확인하고, 아니라면 upgrade 하기. (pyLDA 에러 때문)
# get_ipython().system('pip install pandas==1.2.4')
import pandas as pd
from datetime import datetime
#pd.show_versions()
# pandas           : 1.2.4
# 
# ## ★★ 11장 결정트리분석에서 Pandas를 하위버전으로 설치하였으므로,  최신버전으로 업그레이드 설치한다. 
# ### - Anaconda Prompt 를 [관리자권한으로 실행] 한 후에, 명령어 입력:  pip  install  --upgrade  pandas 
# ###    -> 업그레이드 설치를 적용하기 위해서, Jupyter Notobook을 종료했다가 다시 실행하기 !!
# 13장_감성분석.py 를 실행해 보고 싶은 학생은 다음과 같은 점을 체크한 다음 실행하지 바랍니다.
# (1) pandas 버전을 낮춘다. => pip install pandas==1.4.3
# (2) pyLDAvis 버전을 맞춘다. => pip install pyLDAvis==3.4.0
# (3) kernel을 새로 시작한다.


# (실행참고) 1부, 2부, 3부 단계별("Run Selection 버튼으로) 나누어서 실행한다, 특히 3부는 한 문장씩 실행을 한다.
import pandas as pd
pd.__version__

# ### 한글 UnicoedEncodingError를 방지하기 위해 기본 인코딩을 "utf-8"로 설정
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import time

# ### 경고메시지 표시 안하게 설정하기
import warnings
warnings.filterwarnings(action='ignore')
t1=datetime.now(); print(t1, 'START')
t1=datetime.now(); print(t1, '#### 1')
################################################################
# # 1부. 감성 분류 모델 구축
################################################################
# ## 1. 데이터 수집
# #### 깃허브에서 데이터 파일 다운로드 : https://github.com/e9t/nsmc 
# ## 2. 데이터 준비 및 탐색
# ### 2-1) 훈련용 데이터 준비
# #### (1) 훈련용 데이터 파일 로드
nsmc_train_df = pd.read_csv('./DATA/ratings_train.txt', encoding='utf8', sep='\t', engine='python')
nsmc_train_df.head()

# #### (2) 데이터의 정보 확인
nsmc_train_df.info()

# #### (3) 'document'칼럼이 Null인 샘플 제거
nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]
nsmc_train_df.info()

# #### (4) 타겟 컬럼 label 확인 (0: 부정감성,   1: 긍정감성)
nsmc_train_df['label'].value_counts()

# #### (5) 한글 이외의 문자는 공백으로 변환 (정규표현식 이용)
import re

nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
nsmc_train_df.head()

# ### 2-2) 평가용 데이터 준비
# #### (1) 평가용 데이터 파일 로드
nsmc_test_df = pd.read_csv('./DATA/ratings_test.txt', encoding='utf8', sep='\t', engine='python')
nsmc_test_df.head()

# #### (2) 데이터의 정보 확인
nsmc_test_df.info()

# #### (3) 'document'칼럼이 Null인 샘플 제거
nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]

# #### (4) 타겟 컬럼 label 확인 (0: 부정감성, 1: 긍정감성)
print(nsmc_test_df['label'].value_counts())

# #### (5) 한글 이외의 문자는 공백으로 변환 (정규표현식 이용)

nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", x))

# ## 3. 분석 모델 구축
# ### 3-1) 피처 벡터화 : TF-IDF
# #### (1) 형태소를 분석하여 토큰화 : 한글 형태소 엔진으로 Okt 이용
# get_ipython().system('pip install konlpy')

from konlpy.tag import Okt
okt = Okt()

def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2
# =================
# #### (2) TF-IDF 기반 피처 벡터 생성 : 실행시간 10분 이상 걸립니다 ☺
# =================
print('TF-IDF 기반 피처 벡터 생성 : 실행시간 20분 이상 걸립니다')
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
print(datetime.now())
tfidf.fit(nsmc_train_df['document'])  # 여기서 기다림 
print(datetime.now())
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document']) 
print(datetime.now())
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2
# ### 3-2) 감성 분류 모델 구축 : 로지스틱 회귀를 이용한 이진 분류
# ### - Sentiment Analysis using Logistic Regression
# #### (1) 로지스틱 회귀 기반 분석모델 생성

from sklearn.linear_model import LogisticRegression
SA_lr = LogisticRegression(random_state = 0)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])

# #### (2) 로지스틱 회귀의  best 하이퍼파라미터 찾기

from sklearn.model_selection import GridSearchCV
params = {'C': [1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring='accuracy', verbose=1)

# #### (3) 최적 분석 모델 훈련
print('최적 분석 모델 훈련 ... 1분 이상 소요..'); 
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2
SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df['label'])
print(datetime.now())
print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2
# 최적 파라미터의 best 모델 저장
SA_lr_best = SA_lr_grid_cv.best_estimator_

# ## 4. 분석 모델 평가
# ### 4-1) 평가용 데이터를 이용하여 감성 분석 모델 정확도
# 평가용 데이터의 피처 벡터화 : 실행시간 6분 정도 걸립니다 ☺
print('평가용 데이터의 피처 벡터화 : 실행시간 5분 정도 걸립니다')
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
test_predict = SA_lr_best.predict(nsmc_test_tfidf)
print(datetime.now())
t2=datetime.now(); print(t2, t2-t1, '#### 1'); t1=t2

from sklearn.metrics import accuracy_score
print('감성 분석 정확도 : ', round(accuracy_score(nsmc_test_df['label'], test_predict), 3))

# ### 4-2) 새로운 텍스트에 대한 감성 예측

#st = input('감성 분석할 문장입력 >> ')
st="완전 신파 그 자체네요;;"

# 0) 입력 텍스트에 대한 전처리 수행
st = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(st); print(st)
st = [" ".join(st)]; print(st)

# 1) 입력 텍스트의 피처 벡터화
st_tfidf = tfidf.transform(st)

# 2) 최적 감성분석 모델에 적용하여 감성 분석 평가
st_predict = SA_lr_best.predict(st_tfidf)

# 3) 예측 값 출력하기
if(st_predict== 0):
    print(st , "->> 부정 감성")
else :
    print(st , "->> 긍정 감성")
t2=datetime.now(); print(t2, t2-t1, '#### 2'); t1=t2
################################################################
# # 2부. 감성 분석 수행 
################################################################
# ## 1. 감성 분석할 데이터 수집
# #### - 4장에서 학습한 네이버 API를 이용한 크롤링 프로그램을 이용하여, 네이버 뉴스를 크롤링하여 텍스트 데이터를 수집한다
# ## 2. 데이터 준비 및 탐색
# #### (1) 파일 불러오기
t2=datetime.now(); print(t2, t2-t1, '#### 2-2'); t1=t2
import json
file_name = '정년연장_naver_news' # 정년연장_naver_news.json '코로나_naver_news'

with open('./DATA/'+file_name+'.json', encoding='utf8') as j_f:
    data = json.load(j_f)
print(data)

# #### (2) 분석할 컬럼을 추출하여 데이터 프레임에 저장
data_title =[]
data_description = []

for item in data:
    data_title.append(item['title'])
    data_description.append(item['description'])

data_title
data_description
data_df = pd.DataFrame({'title':data_title, 'description':data_description})

# #### (3) 한글 이외 문자 제거
data_df['title'] = data_df['title'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
data_df['description'] = data_df['description'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
data_df.head()  #작업 확인용 출력

################################################################
# ## 3. 감성 분석 수행
t2=datetime.now(); print(t2, t2-t1, '#### 2-3'); t1=t2
# ### 3-1) 'title'에 대한 감성 분석
# 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
data_title_tfidf = tfidf.transform(data_df['title'])
# 2) 최적 파라미터 학습모델에 적용하여 감성 분석
data_title_predict = SA_lr_best.predict(data_title_tfidf)
# 3) 감성 분석 결과값을 데이터 프레임에 저장
data_df['title_label'] = data_title_predict

# ### 3-2) 'description' 에 대한 감성 분석

# 1) 분석할 데이터의 피처 벡터화 ---<< description >> 분석
data_description_tfidf = tfidf.transform(data_df['description'])
# 2) 최적 파라미터 학습모델에 적용하여 감성 분석
data_description_predict = SA_lr_best.predict(data_description_tfidf)
# 3) 감성 분석 결과값을 데이터 프레임에 저장
data_df['description_label'] = data_description_predict

# ### 3-3)  분석 결과가 추가된 데이터프레임을 CSV 파일 저장

# csv 파일로 저장 ---------------------------------------------
data_df.to_csv('./DATA/'+file_name+'.csv', encoding='euc-kr') 

################################################################
# ## 4. 감성 분석 결과 확인 및 시각화 - 0: 부정감성,   1: 긍정감성
# ### 4-1) 감성 분석 결과 확인
t2=datetime.now(); print(t2, t2-t1, '#### 2-4'); t1=t2
data_df.head()
print(data_df['title_label'].value_counts())
print(data_df['description_label'].value_counts())

# ### 4-2) 결과 저장 : 긍정과 부정을 분리하여 CSV 파일 저장
columns_name = ['title','title_label','description','description_label']
NEG_data_df = pd.DataFrame(columns=columns_name)
POS_data_df = pd.DataFrame(columns=columns_name)

for i, data in data_df.iterrows(): 
    title = data["title"] 
    description = data["description"] 
    t_label = data["title_label"] 
    d_label = data["description_label"] 
    
    # 새로운 행을 만들기 위한 데이터프레임 생성
    new_row = pd.DataFrame([[title, t_label, description, d_label]], columns=columns_name)
    
    if d_label == 0:  # 부정 감성 샘플만 추출
        NEG_data_df = pd.concat([NEG_data_df, new_row], ignore_index=True)
    else:  # 긍정 감성 샘플만 추출
        POS_data_df = pd.concat([POS_data_df, new_row], ignore_index=True)
        
    """
    ↓↓ 교수님 코드는 이럼 ↓↓↓
    if d_label == 0: # 부정 감성 샘플만 추출
        NEG_data_df = pd.concat([NEG_data_df, pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name)],ignore_index=True)#NEG_data_df = NEG_data_df.append(pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name),ignore_index=True)
    else : # 긍정 감성 샘플만 추출
        POS_data_df = pd.concat([POS_data_df, pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name)],ignore_index=True)#POS_data_df = POS_data_df.append(pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name),ignore_index=True)
    """
    
# 파일에 저장.
NEG_data_df.to_csv('./DATA/'+file_name+'_NES.csv', encoding='euc-kr') 
POS_data_df.to_csv('./DATA/'+file_name+'_POS.csv', encoding='euc-kr') 

len(NEG_data_df), len(POS_data_df)

# ### 4-3)  감성 분석 결과 시각화 : 바 차트
# #### (1) 명사만 추출하여 정리하기
# #### - 긍정 감성의 데이터에서 명사만 추출하여 정리 
POS_description = POS_data_df['description']

POS_description_noun_tk = []
for d in POS_description:
    POS_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출

print(POS_description_noun_tk)  #작업 확인용 출력

POS_description_noun_join = []
for d in POS_description_noun_tk:
    d2 = [w for w in d if len(w) > 1] #길이가 1인 토큰은 제외
    POS_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성
print(POS_description_noun_join)  #작업 확인용 출력

# #### - 부정 감성의 데이터에서 명사만 추출하여 정리 
NEG_description = NEG_data_df['description']
NEG_description_noun_tk = []
NEG_description_noun_join = []

for d in NEG_description:
    NEG_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출
    
for d in NEG_description_noun_tk:
    d2 = [w for w in d if len(w) > 1]  #길이가 1인 토큰은 제외
    NEG_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성
print(NEG_description_noun_join)  #작업 확인용 출력

# #### (2) dtm 구성 : 단어 벡터 값을 내림차순으로 정렬
# #### - 긍정 감성 데이터에 대한 dtm 구성, dtm을 이용하여 단어사전 구성 후 내림차순 정렬
POS_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
POS_dtm = POS_tfidf.fit_transform(POS_description_noun_join)

POS_vocab = dict() ; # dir(POS_tfidf)
for idx, word in enumerate(POS_tfidf.get_feature_names_out()):
    POS_vocab[word] = POS_dtm.getcol(idx).sum()
    
POS_words = sorted(POS_vocab.items(), key=lambda x: x[1], reverse=True)
POS_words  #작업 확인용 출력

# #### - 부정 감성 데이터의 dtm 구성, dtm을 이용하여 단어사전 구성 후 내림차순 정렬
NEG_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
NEG_dtm = NEG_tfidf.fit_transform(NEG_description_noun_join)

NEG_vocab = dict() 

for idx, word in enumerate(NEG_tfidf.get_feature_names_out()):
    NEG_vocab[word] = NEG_dtm.getcol(idx).sum()
    
NEG_words = sorted(NEG_vocab.items(), key=lambda x: x[1], reverse=True)
NEG_words   #작업 확인용 출력
t2=datetime.now(); print(t2, t2-t1, '#### 2-4'); t1=t2
# #### (3) 단어사전의 상위 단어로 바 차트 그리기

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#fm.get_fontconfig_fonts() # dir(fm)
font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

max = 15  #바 차트에 나타낼 단어의 수 

plt.bar(range(max), [i[1] for i in POS_words[:max]], color="blue")
plt.title("긍정 뉴스의 단어 상위 %d개" %max, fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max), [i[0] for i in POS_words[:max]], rotation=70)
plt.show()

plt.bar(range(max), [i[1] for i in NEG_words[:max]], color="red")
plt.title("부정 뉴스의 단어 상위 %d개" %max, fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max), [i[0] for i in NEG_words[:max]], rotation=70)
plt.show()

data_df.to_csv("datadf") # 결과 저장 후 3부에서 사용
t2=datetime.now(); print(t2, t2-t1, '#### 2-4'); t1=t2

input('3부. 토픽모델링 계속할까요? (y)')
################################################################
# # 3부. 토픽모델링 : LDA 기반 토픽 모델링
################################################################
#최초 한번만 pyLDAvis 설치(conda 사용 권장)
# (1) pandas 버전을 낮춘다. => pip install pandas==1.4.3
# (2) pyLDAvis 버전을 맞춘다. => pip install pyLDAvis==3.4.0
#     (설치 Anaconda Powershell Prompt에서) 
#     conda install -c conda-forge pyLDAvis==3.4.0
# (3) kernel을 새로 시작한다.
# (4)  1부, 2부, 3부 단계별 나누어서 실행한다, 특히 3부는 한 문장씩 실행을 한다.

from datetime import datetime
t2=datetime.now(); print(t2, t2-t1, '#### 3'); t1=t2
import pandas as pd
from konlpy.tag import Okt
okt = Okt()
file_name = '정년연장_naver_news' # 정년연장_naver_news.json '코로나_naver_news'
# ## 1. 데이터 준비 
# ### 1-1) 'description' 컬럼 추출
t2=datetime.now(); print(t2, t2-t1, '#### 3-1'); t1=t2
data_df=pd.read_csv("datadf")
description = data_df['description']

# ### 1-2) 형태소 토큰화 : 명사만 추출

description_noun_tk = []
for d in description:
    description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출
description_noun_tk2 = []
for d in description_noun_tk:
    item = [i for i in d if len(i) > 1]  #토큰의 길이가 1보다 큰 것만 추출
    description_noun_tk2.append(item)
#print(description_noun_tk2)

# ## 2. LDA 토픽 모델 구축
# ### 2-1) LDA 모델의 입력 벡터 생성 
# 최초 한번만 설치
# get_ipython().system('pip install gensim   ')
t2=datetime.now(); print(t2, t2-t1, '#### 3-2'); t1=t2
import gensim
import gensim.corpora as corpora

# #### (1) 단어 사전 생성
dictionary = corpora.Dictionary(description_noun_tk2)
print(dictionary[1])  #작업 확인용 출력  감염증?

# #### (2) 단어와 출현빈도(count)의 코퍼스 생성
corpus = [dictionary.doc2bow(word) for word in description_noun_tk2]
#print(corpus) #작업 확인용 출력

# ### 2-2) LDA 모델 생성 및 훈련 
k = 4  #토픽의 개수 설정
lda_model = gensim.models.ldamulticore.LdaMulticore(corpus, iterations = 12, num_topics = k, id2word = dictionary, passes = 1, workers = 10)

# ## 3. LDA 토픽 분석 결과 시각화
# ### 3-1) 토픽 분석 결과 확인
t2=datetime.now(); print(t2, t2-t1, '#### 3-3'); t1=t2
print("#### lda_model.print_topics")
print(lda_model.print_topics(num_topics = k, num_words = 15))

# ### 3-2) 토픽 분석 결과 시각화 : pyLDAvis
print("#### pyLDAvis")
import pyLDAvis
print(pyLDAvis.__version__) # 3.4.0
import pyLDAvis.gensim_models
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
lda_vis
pyLDAvis.display(lda_vis)
pyLDAvis.save_html(lda_vis, './DATA/'+file_name+"_vis.html")
t2=datetime.now(); print(t2, t2-t1, '#### END'); t1=t2
# 결과 ./DATA/코로나_naver_news_vis.html


"""
pip list
"""



