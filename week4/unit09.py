#!/usr/bin/env python
# coding: utf-8

# In[4]:


import csv
f = open('gender.csv')
data = csv.reader(f)
m = []
f = []
name = input('찾고 싶은 지역의 이름을 알려주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        for i in row[3:104] :
            m.append(-int(i))
        for i in row[106:] :
            f.append(int(i))
print(len(m), len(f))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.title(name + ' 지역의 남녀 성별 인구 분포')
plt.barh(range(101), m, label = '남성')
plt.barh(range(101), f, label = '여성')
plt.legend()
plt.show()


# In[2]:


import csv
f = open('gender.csv')
data = csv.reader(f)
m = []
f = []
name = input('찾고 싶은 지역의 이름을 알려주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        for i in row[3:104] :
            m.append(-int(i))
        for i in row[106:] :
            f.append(int(i))
        break  ## 1개만 저장하고 빠져나가기....

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.title(name + ' 지역의 남녀 성별 인구 분포')
plt.barh(range(101), m, label = '남성')
plt.barh(range(101), f, label = '여성')
plt.legend()
plt.show()


# In[5]:


import matplotlib.pyplot as plt
plt.pie([10,20])
plt.show()


# In[6]:


import matplotlib.pyplot as plt
size = [2441, 2312, 1031, 1233]
plt.axis('equal')
plt.pie(size)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')
size = [2441, 2312, 1031, 1233]
label = ['A형','B형','AB형', 'O형']
plt.axis('equal')
plt.pie(size, labels = label, autopct = '%.1f%%')   # 비율 및 범례 추가
plt.legend()
plt.show()


# In[8]:


import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')
size = [2441, 2312, 1031, 1233]
label = ['A형','B형','AB형', 'O형']
color = ['darkmagenta', 'deeppink', 'hotpink', 'pink']
plt.axis('equal')
plt.pie(size, labels = label, autopct = '%.1f%%', explode = (0,0,0.1,0), colors = color)
plt.legend()
plt.show()


# In[7]: 색상표 확인 가능♡


import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]
n = len(sorted_names)
ncols = 4
nrows = n // ncols
fig, ax = plt.subplots(figsize=(9, 8), dpi = 300)

# Get height and width
X, Y = fig.get_dpi() * fig.get_size_inches()
h = Y / (nrows + 1)
w = X / ncols

for i, name in enumerate(sorted_names):
    row = i % nrows
    col = i // nrows
    y = Y - (row * h) - h
    xi_line = w * (col + 0.05)
    xf_line = w * (col + 0.25)
    xi_text = w * (col + 0.3)
    ax.text(xi_text, y, name, fontsize=(10),
            horizontalalignment='left',
            verticalalignment='center')
    ax.hlines(y + h * 0.1, xi_line, xf_line,
              color=colors[name], linewidth=(6))
ax.set_xlim(0, X)
ax.set_ylim(0, Y)
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)
plt.show()


# In[13]:


import csv
f = open('gender.csv')
data = csv.reader(f)
size = []
name = input('찾고 싶은 지역의 이름을 알려주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        m = 0
        f = 0
        for i in range(101) :
            m += int(row[i+3])
            f += int(row[i+106])
        break
size.append(m)
size.append(f)

import matplotlib.pyplot as plt
plt.rc('font', family ='Malgun Gothic')
color = ['crimson', 'darkcyan']
plt.axis('equal')
plt.pie(size, labels = ['남','여'], autopct ='%.1f%%', colors = color, startangle =90)
plt.title(name + ' 지역의 남녀 성별 비율')
plt.show()


# In[ ]:




