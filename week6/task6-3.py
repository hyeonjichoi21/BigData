import requests
from bs4 import BeautifulSoup
import pandas as pd

# 페이지 요청 및 파싱 
url = "https://www.weather.go.kr/w/observation/land/city-obs.do"
headers = {
    "User-Agent": "Mozilla/5.0"
}
res = requests.get(url, headers=headers)
res.encoding = "utf-8"
soup = BeautifulSoup(res.text, "html.parser")

# 테이블 찾기
table = soup.find("table", {"id": "weather_table"})

data = []
if table:
    for row in table.tbody.find_all("tr"):
        # 지점명(sido-gu)(th), 나머지는 td
        th = row.find("th")
        tds = row.find_all("td")
        if th and len(tds) >= 11:
            city = th.get_text(strip=True)
            temp = tds[4].get_text(strip=True)      # 6번째 td: 현재기온
            humid = tds[9].get_text(strip=True)    # 11번째 td: 습도
            # 결측치 처리
            if temp and humid:
                data.append([city, temp, humid])

df = pd.DataFrame(data, columns=["sido-gu", "온도", "습도"])
print(df)
print(df.info())
