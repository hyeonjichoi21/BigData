# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:31:46 2025

@author: gusdk
"""

import os
import sys
import urllib.request
client_id = "W4LY21H9WKgd08OF3TId"
client_secret = "5zOcEFXspg"
encText = urllib.parse.quote("NCT WISH")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    myresult = response_body.decode('utf-8')
    print(myresult)
else:
    print("Error Code:" + rescode)