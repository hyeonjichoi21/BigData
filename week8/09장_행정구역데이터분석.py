#!/usr/bin/env python
# coding: utf-8

# # 9장. 지리 정보 분석 (2) 행정구역별 데이터 분석 + 블록맵
# # 1. 데이터 준비 및 탐색
# In[1]:
import pandas as pd
pd.set_option('mode.chained_assignment',  None) # Warning 방지용
import numpy as np

data = pd.read_csv('./DATA/공공보건의료기관현황.csv', index_col=0, encoding='CP949', engine='python')
data.head() #작업내용 확인용 출력

# In[2]:
## 주소에서 시도, 군구 정보 분리
addr = pd.DataFrame(data['주소'].apply(lambda v: v.split()[:2]).tolist(),columns=('시도', '군구'))
addr.head()  #작업내용 확인용 출력

# ## 1) 시도 이름 확인하기
# In[3]:
addr['시도'].unique()

# ### #잘못된 시도 이름 수정 : 창원시, 경산시, 천안시
# #### (1) '창원시'를 찾아서 '경상남도 창원시'로 수정
# In[4]:
addr[addr['시도'] == '창원시']

# In[5]:
## 표준 행정구역 이름으로 수정 : 창원시-> 경상남도 창원시
addr.iloc[27] = ['경상남도', '창원시']
addr.iloc[31] = ['경상남도', '창원시']

# In[6]:
addr.iloc[27]

# In[7]:
addr.iloc[31]

# #### (2) '경산시'를 찾아서 '경상북도 경산시'로 수정
# In[8]:
addr[addr['시도'] == '경산시']

# In[9]:
addr.iloc[47] = ['경상북도', '경산시']
addr.iloc[47]

# #### (3) '천안시'를 찾아서 '충청남도 천안시'로 수정
# In[10]:
addr[addr['시도'] == '천안시']

# In[11]:
## 표준 행정구역 이름으로 수정 : 천안시-> 충청남도 천안시
addr.iloc[209] = ['충청남도', '천안시']
addr.iloc[210] = ['충청남도', '천안시']

# In[12]:
addr.iloc[209]

# In[13]:
addr.iloc[210]

# In[14]:
# 작업 결과 확인하기
addr['시도'].unique()

# ### # 시도 이름을 표준이름으로 수정
# In[15]:
## 표준 행정구역 이름으로 수정 :  경기 -> 경기도, 경남 -> 경상남도, ...
addr_aliases = {'경기':'경기도', '경남':'경상남도', '경북':'경상북도', '충북':'충청북도', '서울시':'서울특별시', '부산특별시':'부산광역시', '대전시':'대전광역시', '충남':'충청남도', '전남':'전라남도', '전북':'전라북도'}

# In[16]:
addr['시도']= addr['시도'].apply(lambda v: addr_aliases.get(v, v))

# In[17]:
# 작업 결과 확인하기
addr['시도'].unique()

# ## 2) 군구 이름 확인하기 
# In[18]:
addr['군구'].unique()

# ### # 잘못된 군구 이름 수정하기 
# #### (1) '아란13길' 을 '제주특별자치도'  '제주시'로 수정
# In[19]:
addr[addr['군구'] == '아란13길']

# In[20]:
addr.iloc[75] = ['제주특별자치도', '제주시']
addr.iloc[75]

# In[21]:
addr['군구'].unique()

# ## 3) 행정구역별 공공보건의료기관의 수 구하기
# ### (1) '시도' 와 '군구' 컬럼 결합하기
# In[22]:
addr['시도군구'] = addr.apply(lambda r: r['시도'] + ' ' + r['군구'], axis=1)
addr.head() #작업 확인용 출력

# In[23]:
addr['count'] = 0  # 의료기관수 합계를 저장할 컬럼 만들기
addr.head() #작업 확인용 출력

# ### (2) '시도군구' 를 기준으로 그룹을 만들고, 그룹별 의료기관수 합계 구하기
# In[24]:
addr_group =pd.DataFrame(addr.groupby(['시도', '군구', '시도군구'], as_index=False).count())
addr_group.head()  #작업 확인용 출력

# ### (3) 데이터 병합에 사용할 인덱스 설정하기
# In[25]:
addr_group = addr_group.set_index("시도군구") # set_index: "시도군구"가 index로 바뀜
addr_group.head()   #작업 확인용 출력

# ## 4) 행정구역 인구수 컬럼 추가하기
# ####   (1) 행정구역 이름 데이터 불러오기 : 행정구역_시군구_별__성별_인구수_2.xlsx
# In[26]:
population = pd.read_excel('./DATA/행정구역_시군구_별__성별_인구수_2.xlsx')
population.head()    #작업 확인용 출력

# In[27]:
population = population.rename(columns = {'행정구역(시군구)별(1)': '시도', '행정구역(시군구)별(2)': '군구'}) #컬럼이름 변경
population.head()  #작업 확인용 출력

# ####   (2) 병합에 사용할 공통 컬럼 '시도군구' 추가하기
# In[28]:
# '군구' 컬럼에서 공백 제거하기
for element in range(0,len(population)):
      population['군구'][element] = population['군구'][element].strip()

# In[29]:
# '시도'와 '군구'를 연결하여 '시도군구' 컬럼 추가
population['시도군구']= population.apply(lambda r: r['시도'] + ' ' + r['군구'], axis=1)
population.head()  #작업 확인용 출력

# ##### -  필요없는  '소계' 행 삭제
# In[30]:
population = population[population.군구 != '소계']
population.head()  #작업 확인용 출력

# ####   (3) 병합의 기준이 될 인덱스를 '시도군구'로 설정
# In[31]:
population = population.set_index("시도군구")
population.head()  #작업 확인용 출력

# ## 5) '의료기관' 데이터프레임과 '시도군구별 인구수' 데이터프레임 병합하기 
# In[32]:
addr_population_merge = pd.merge(addr_group,population,  how='inner',  left_index=True, right_index=True)
addr_population_merge.head()   #작업 확인용 출력

# ##### -필요한 컬럼만 추출하기
# In[33]:
local_MC_Population = addr_population_merge[['시도_x', '군구_x',  'count', '총인구수 (명)']]
local_MC_Population.head()   #작업 확인용 출력  

# In[34]:
#컬럼이름 변경
local_MC_Population = local_MC_Population.rename(columns = {'시도_x': '시도', '군구_x': '군구','총인구수 (명)': '인구수' })
local_MC_Population.head()  #작업 확인용 출력

# ## 6) 시도군구의 인구대비 의료기관수 비율 구하기
# In[35]:
MC_count = local_MC_Population['count']
local_MC_Population['MC_ratio'] = MC_count.div(local_MC_Population['인구수'], axis=0)*100000
local_MC_Population.head()   #작업 확인용 출력

# # 2. 분석 모델 구축 및 시각화
# ##  1) 바 차트 그리기
# In[36]:
from matplotlib import pyplot as plt
from matplotlib import rcParams, style
style.use('ggplot')

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# ####   (1) 행정구역별 공공보건의료기관수에 대한 바 차트 
# In[37]:
MC_ratio = local_MC_Population[['count']]
MC_ratio = MC_ratio.sort_values('count', ascending = False)
plt.rcParams["figure.figsize"] = (25,5)
MC_ratio.plot(kind='bar', rot=90)
plt.show()

# ####   (2) 행정구역별 인구수 대비 공공보건의료기관 비율에 대한 바 차트 
# In[38]:
MC_ratio = local_MC_Population[['MC_ratio']]
MC_ratio = MC_ratio.sort_values('MC_ratio', ascending = False)
plt.rcParams["figure.figsize"] = (25,5)
MC_ratio.plot(kind='bar', rot=90)
plt.show()

# ## 2) 블록맵 시각화
# ###   (1) 블록맵 데이터 파일 열기
# In[39]:
import os
path = os.getcwd()

# In[40]:
data_draw_korea = pd.read_csv(path+'./DATA/data_draw_korea.csv', index_col=0, encoding='UTF-8', engine='python')
data_draw_korea.head()   #작업 확인용 출력

# ###   (2) 블록맵 데이터 파일에 '시도군구' 컬럼 만들기
# In[41]:
data_draw_korea['시도군구']= data_draw_korea.apply(lambda r: r['광역시도'] + ' ' + r['행정구역'], axis=1)
data_draw_korea.head()  #작업 확인용 출력

# ###   (3) 블록맵 데이터에서 병합에 사용할 '시도군구' 컬럼을 인덱스로 설정하기
# In[42]:
data_draw_korea = data_draw_korea.set_index("시도군구")
data_draw_korea.head()  #작업 확인용 출력

# ###   (4) 블록맵 데이터프레임과 local_MC_Population을 병합하기
# In[43]:
data_draw_korea_MC_Population_all = pd.merge(data_draw_korea,local_MC_Population,  how='outer',  left_index=True, right_index=True)
data_draw_korea_MC_Population_all.head()

# ###   (5) 한국지도의 블록맵 경계선 좌표를 리스트로 생성 
# In[44]:
BORDER_LINES = [
    [(3, 2), (5, 2), (5, 3), (9, 3), (9, 1)], # 인천
    [(2, 5), (3, 5), (3, 4), (8, 4), (8, 7), (7, 7), (7, 9), (4, 9), (4, 7), (1, 7)], # 서울
    [(1, 6), (1, 9), (3, 9), (3, 10), (8, 10), (8, 9),
     (9, 9), (9, 8), (10, 8), (10, 5), (9, 5), (9, 3)], # 경기도
    [(9, 12), (9, 10), (8, 10)], # 강원도
    [(10, 5), (11, 5), (11, 4), (12, 4), (12, 5), (13, 5),
     (13, 4), (14, 4), (14, 2)], # 충청남도
    [(11, 5), (12, 5), (12, 6), (15, 6), (15, 7), (13, 7),
     (13, 8), (11, 8), (11, 9), (10, 9), (10, 8)], # 충청북도
    [(14, 4), (15, 4), (15, 6)], # 대전시
    [(14, 7), (14, 9), (13, 9), (13, 11), (13, 13)], # 경상북도
    [(14, 8), (16, 8), (16, 10), (15, 10),
     (15, 11), (14, 11), (14, 12), (13, 12)], # 대구시
    [(15, 11), (16, 11), (16, 13)], # 울산시
    [(17, 1), (17, 3), (18, 3), (18, 6), (15, 6)], # 전라북도
    [(19, 2), (19, 4), (21, 4), (21, 3), (22, 3), (22, 2), (19, 2)], # 광주시
    [(18, 5), (20, 5), (20, 6)], # 전라남도
    [(16, 9), (18, 9), (18, 8), (19, 8), (19, 9), (20, 9), (20, 10)], # 부산시
]

# ###   (6) 블록맵에서 블록에 해당 데이터를 매핑하여 색을 표시하는 함수
# In[45]:
def draw_blockMap(blockedMap, targetData, title, color ):
    whitelabelmin = (max(blockedMap[targetData]) - min(blockedMap[targetData])) * 0.25 + min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(8, 13))
    plt.title(title)
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=color, edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'
    
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. (중구, 서구)
        if row['광역시도'].endswith('시') and not row['광역시도'].startswith('세종'):
            dispname = '{}\n{}'.format(row['광역시도'][:2], row['행정구역'][:-1])
            if len(row['행정구역']) <= 2:
                dispname += row['행정구역'][-1]
        else:
            dispname = row['행정구역'][:-1]

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 9.5, 1.5
        else:
            fontsize, linespacing = 11, 1.2

        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)
    
    # 시도 경계 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=4)

    plt.gca().invert_yaxis()
    #plt.gca().set_aspect(1)
    plt.axis('off')
    
    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    
    plt.savefig('./DATA/' + 'blockMap_' + targetData + '.png')
    plt.show()      

# ###   (7) 함수를 호출하여 블록맵 생성하기
# #####     - 행정구역별 인구에 대한 의료기관 수에 대한 블록맵
# In[46]:
draw_blockMap(data_draw_korea_MC_Population_all, 'count', '행정구역별 공공보건의료기관 수', 'Blues')

# #####     - 행정구역별 인구에 대한 의료기관 비율에 대한 블록맵
# In[47]:
draw_blockMap(data_draw_korea_MC_Population_all, 'MC_ratio', '행정구역별 인구수 대비 공공보건의료기관 비율', 'Reds' )
