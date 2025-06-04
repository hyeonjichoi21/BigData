# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:36:03 2023

@author: USER
"""

text="난 우리영화를 love 사랑합니다....^^"
import re
newstr=re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", text) # ㄱ부터 힣까지 한글 전체
print(newstr)

from konlpy.tag import Okt #한국어 정보처리를 위한 파이썬 패키지
okt = Okt() 
tokens = okt.morphs(text)
print(tokens)
# ['난', '우리', '영화', '를', '사랑', '합니다']

#********* 명사 찾기 okt.noun()
no=okt.nouns(newstr)
print(no)

# 명사만 뽑아줌 ~~ 
