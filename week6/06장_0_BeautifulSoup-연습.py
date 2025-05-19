from bs4 import BeautifulSoup

html = '<h1 id="title">한빛출판네트워크</h1><div class="top"><ul class="menu"><li> \
<a href=http://www.hanbit.co.kr/member/login.html class="login">로그인 </a></li></ul><ul class="brand"> \
<li><a href="http://www.hanbit.co.kr/media/">한빛미디어 \
<li><a href="http://www.hanbit.co.kr/academy/">한빛아카데미</a></li></ul></div>'

soup = BeautifulSoup(html, 'html.parser')
print(soup.prettify())

soup.h1
tag_h1 = soup.h1
tag_h1
tag_div = soup.div
tag_div
tag_ul = soup.ul
tag_ul
tag_li = soup.li
tag_li
tag_a = soup.a
tag_a
tag_ul_all = soup.find_all("ul")
tag_ul_all
tag_li_all = soup.find_all("li")
tag_li_all
tag_a_all = soup.find_all("a")
tag_a_all
tag_a.attrs
tag_a['href']
tag_a['class']
tag_ul_2 = soup.find('ul', attrs={'class':'brand'})
tag_ul_2
title = soup.find(id="title")
title
title.string
li_list = soup.select("div>ul.brand>li")
li_list
for li in li_list:
    print(li.string)
    

