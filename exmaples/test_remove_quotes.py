import re
import string

sentence = "+今天=是！2021!   年/8月?1,7日★.---《七夕节@》：让我*们出门（#@）去“感受”夏天的荷尔蒙！"
sentenceClean = []
# method 1
remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
string1 = re.sub(remove_chars, "", sentence)
sentenceClean.append(string1)

# method 2
punct = str.maketrans({key:"" for key in string.punctuation})
# 这里的string中包含的标点符号不是很全
# string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 都是英文字符下的标点
string2 = sentence.translate(punct)
sentenceClean.append(string2)

# method 3
string3 = "".join(re.findall(r'\b\w+\b',sentence))
# 正则表达式中\b可以简单理解为单词的边界（指的是字母数字和非字母数字的边界），\w表示字母数字下划线，
#'\b\w+\b'在这道题中就能做到匹配一个单词，re.findall是将全部的单词找出来
sentenceClean.append(string3)

# method 4
string4 = re.sub('\W*', '', sentence) # 把非单词字符全部替换为空，恰好与\w相反
sentenceClean.append(string4)


print(sentence)
print(sentenceClean)
'''
以下的结果有一些细微的差别，可以自行对比查找下原因。
result:
+今天=是！2021!   年/8月?1,7日★.---《七夕节@》：让我*们出门（#@）去“感受”夏天的荷尔蒙！
['今天是2021   年8月17日七夕节让我们出门去感受夏天的荷尔蒙', 
'今天是！2021   年8月17日★《七夕节》：让我们出门（）去“感受”夏天的荷尔蒙！', 
'今天是2021年8月17日七夕节让我们出门去感受夏天的荷尔蒙',
'今天是2021年8月17日七夕节让我们出门去感受夏天的荷尔蒙']
'''
