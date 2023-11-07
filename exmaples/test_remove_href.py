import re

string1 = '我大航海时代获得很好的或多或少的基督教https://weibo.com/1699432410/GC59cqsyF们'
string2 = '我http://qiye.tianya.cn//blog/infoReader3.aspxblogID=1776&ComID=12&infoType=1们'
string3 = '日本美少年系列！要表白的人太多了，都超帅的！！！[色] up:林中人ruirui http://t.cn/RHamEuQ'
results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
string1 = re.sub(results, '', string1)
print(string1)
string2 = re.sub(results, '', string2)
print(string2)

string3= re.sub(results, '', string3)
print(string3)