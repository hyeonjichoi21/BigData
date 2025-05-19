# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:15:23 2025

@author: gusdk
"""

# Python3 샘플 코드 # 1시간 뒤 할 때에는, 마이페이지>신청>인증키 디코딩 키 들어가면 됨


import requests

# 일단 교수님 코드로 함 

url = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
params ={'serviceKey' : '90M3jR0YcAkVYvlEONRII3u/fKqWX/gTWlKh7+k3u9qPH88Zsh0zEZKTOIHTvg+SRLJQjMDg99+qR/NRqb53Dg==', 
         'YM' : '201201', 'NAT_CD' : '112', 'ED_CD' : 'E' }

response = requests.get(url, params=params)
print("결과")
print(response.content)
