#!/usr/bin/env python
# coding: utf-8

# # 9장. 지리 정보 분석 (1) 주소데이터분석+맵
# # 1. 데이터 수집
# ### 데이터 파일 읽어오기
# In[1]:
import pandas as pd
CB = pd.read_csv('./DATA/CoffeeBean.csv', encoding='CP949', index_col=0, header=0, engine='python')
CB.head()  #작업 내용 확인용 출력

# # 2. 데이터 준비 및 탐색
# ## 시/도 행정구역 이름 정규화
# In[2]:
addr = []
for address in CB.address:
    addr.append(str(address).split())

#작업 내용 확인용 출력
print('데이터 개수 : %d' % len(addr)) 
addr  

# ### - addr에서 행정구역 표준 이름이 아닌것 수정하기
# In[3]:
addr2 = []
# addr에서 행정구역 표준 이름이 아닌것 수정하기
for i in range(len(addr)):
    if addr[i][0] == "서울": addr[i][0]="서울특별시"
    elif addr[i][0] == "서울시": addr[i][0]="서울특별시"
    elif addr[i][0] == "부산시": addr[i][0]="부산광역시"
    elif addr[i][0] == "인천": addr[i][0]="인천광역시"
    elif addr[i][0] == "광주": addr[i][0]="광주광역시"
    elif addr[i][0] == "대전시": addr[i][0]="대전광역시"
    elif addr[i][0] == "울산시": addr[i][0]="울산광역시"    
    elif addr[i][0] == "세종시": addr[i][0]="세종특별자치시"
    elif addr[i][0] == "경기": addr[i][0]="경기도"
    elif addr[i][0] == "충북": addr[i][0]="충청북도"
    elif addr[i][0] == "충남": addr[i][0]="충청남도"
    elif addr[i][0] == "전북": addr[i][0]="전라북도"
    elif addr[i][0] == "전남": addr[i][0]="전라남도"
    elif addr[i][0] == "경북": addr[i][0]="경상북도"
    elif addr[i][0] == "경남": addr[i][0]="경상남도"
    elif addr[i][0] == "제주": addr[i][0]="제주특별자치도"
    elif addr[i][0] == "제주도": addr[i][0]="제주특별자치도"
    elif addr[i][0] == "제주시": addr[i][0]="제주특별자치도"                                
       
    addr2.append(' '.join(addr[i]))  

addr2 #작업 내용 확인용 출력

# In[4]:
addr2 = pd.DataFrame(addr2, columns=['address2'])
addr2 #작업 내용 확인용 출력

# In[5]:
CB2 = pd.concat([CB, addr2],  axis=1 ) # axis=1 이면 두 개를 옆(가로)으로 붙인거다
CB2.head()  #작업 내용 확인용 출력

# In[6]:
CB2.to_csv('./DATA/CoffeeBean_2.csv',encoding='CP949', index = False)

# # 3. 데이터 모델링
# ### - 지도 정보 시각화 라이브러리 설치 및 임포트하기
# In[ ]:
#get_ipython().system('pip install folium')

# In[7]:
import folium
# ### 1) 숭례문 좌표를 사용하여 지도 객체 테스트하기

# In[8]:
map_osm = folium.Map(location=[37.560284, 126.975334], zoom_start = 16)

# In[9]:
map_osm.save('./DATA/map.html')

# ### 2) 정리해둔 CoffeeBean_2.csv 파일 로드
# In[10]:
CB_file = pd.read_csv('./DATA/CoffeeBean_2.csv',encoding='cp949',  engine='python')
CB_file.head() #작업 내용 확인용 출력

# ### 3) 오픈 소프트웨어 Geocoder-Xr을 사용하여 구한 GPS 좌표 파일 로드
# In[11]:
CB_geoData = pd.read_csv('./DATA/CB_geo.shp_2.csv',encoding='cp949',  engine='python')
len(CB_geoData) #확인용 출력

# In[12]:
map_CB = folium.Map(location=[37.560284, 126.975334], zoom_start = 15) # 남대문

# 덕성여자대학교 영근터 위치: 37.651105, 127.015775 (구글 지도로 가져옴)
# 구글 지도 -> 마우스 우클릭 -> 이 곳이 궁금한가요? -> 위.경도 복사
map_CB = folium.Map(location=[37.651105, 127.015775], zoom_start = 15)


# In[13]:
for i, store in CB_geoData.iterrows():   
    folium.Marker(location=[store['위도'], store['경도']], popup= store['store'], icon=folium.Icon(color='red', icon='star')).add_to(map_CB)

# In[14]:
map_CB.save('./DATA/map_CB.html') # GPS 찾는 것은 구글이 젤 쉬움

# In[15]:# 크롬이 디폴트로 잡혀있어야 실행됨 ( 난 엣지가 디폴트라 실행 안 됨)
import webbrowser
webbrowser.open('C:/BigData/DATA/map_CB.html')
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
webbrowser.get(chrome_path).open('C:/BigData/DATA/map_CB.html')

