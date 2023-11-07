from snownlp import SnowNLP

text1 = '这个东西不错'
text2 = '这个东西很垃圾'

text3 = ('芯片登记主人信息领养绝育狂犬免疫；禁止繁殖买卖；遗弃虐待不文明列入养宠黑名单；非法经营猫狗肉店全部关停；'
         '世界第二大经济体的中国还狂犬病频发[蜡烛]当街残忍捕杀流浪狗惹众怒负面事件频发[伤心]永远是团理不清的乱麻[笑cry]全国各地很多民间流浪动物救助站举步维艰的承担着整个社会的责任')
text4='日本美少年系列！要表白的人太多了，都超帅的！！！[色] up:林中'
s1 = SnowNLP(text1)
s2 = SnowNLP(text2)
s3 = SnowNLP(text3)
s4 = SnowNLP(text4)

print(s1.sentiments, s2.sentiments,s3.sentiments,s4.sentiments)
# result 0.8623218777387431 0.21406279508712744
