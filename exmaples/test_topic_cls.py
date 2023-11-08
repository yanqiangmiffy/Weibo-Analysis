# -*- coding: utf-8 -*-
import joblib
import jieba
tfidf_mdoel = joblib.load('../models/tfidf_model.joblib')
cls_model = joblib.load('../models/classifier.joblib')
le_model = joblib.load('../models/le.joblib')

# text = '''
# 马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。
# 7月31日下午6点，国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没有受到任何干扰。
# 下午6点，当球队抵达训练场时，大雨已经下了几个小时，而且丝毫没有停下来的意思。抱着试一试的态度，球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们，国奥队决定中止当天的训练，全队立即返回酒店。
# 在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在沈阳最后一周的训练，国奥队首先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、控制感冒等疾病的出现被队伍放在了相当重要的位置。
# 而抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在长春患上了感冒，因此也没有参加29日跟塞尔维亚的热身赛。队伍介绍说，冯萧霆并没有出现发烧症状，但为了安全起见，这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有了冯萧霆这个例子，因此国奥队对雨中训练就显得特别谨慎，主要是担心球员们受凉而引发感冒，造成非战斗减员。
# 而女足队员马晓旭在热身赛中受伤导致无缘奥运的前科，也让在沈阳的国奥队现在格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表示。
# 从长春到沈阳，雨水一路伴随着国奥队，“也邪了，我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种事情。”一位国奥球员也对雨水的“青睐”有些不解。
# '''
text='''
沪基指半日涨2.01% 两市封基近乎全线上扬全景网2月6日讯 沪深基金指数周五早盘大幅收高，深基指收复3000点大关，两市封闭式基金近乎全线上扬，因外围市场提振A股呈现普涨格局。沪基指开盘于2914.88点，高开6.92点，午盘收盘于2966.53点，涨48.57点或2.01%；深基指开盘于2985.63点，高开0.39点，午盘收盘于3035.18点，涨49.94点或1.67%。两市基金半日成交金额为13.75亿元，较上个交易日放大接近一成，成交量为1097.2万手。从盘面上看，开盘交易的32只封闭式基金中，31只上涨，仅1只下跌。基金通乾半日上涨2.93%，领涨封基，其他涨幅较大的有基金普丰上涨2.41%，基金兴华上涨2.19%，基金兴和上涨2.14%，基金同益上涨2.04%，瑞福进取上涨1.97%，基金景福上涨1.93%；富国天丰再度成为唯一下跌的封基，跌幅为0.30%。LOF场内交易方面，开盘交易的27只基金午盘全线上涨。嘉实300上涨2.17%，涨幅第一，其他靠前的有融通巨潮上涨1.88%，鹏华动力上涨1.71%，景顺资源上涨1.57%，大摩资源上涨1.45%，招商成长上涨1.45%，融通领先上涨1.41%；万家公用涨幅最小，半日涨0.17%。ETF午盘全线上涨，且涨幅均在2%上下。上证50ETF涨2.25%，报1.636元；上证180ETF涨2.70%，报5.026元；上证红利ETF涨2.15%，报1.807元；深证100ETF涨2.31%，报2.344元；中小板ETF涨1.94%，报1.573元。（全景网/雷鸣）
财经	牛年第一月 开基抬头券商集合理财掉队每经记者 于春敏在金融危机的淫威之下，2008年，全球资本市场均经历了一番血雨腥风的洗礼，进入2009年，对大多数国家的股市而言，仍然是一片愁云惨淡，看不到放晴迹象，然而，对于A股市场而言，却大有新年新气象、风景这边独好之势。数据显示，在刚刚过去的1月份，A股市场一枝独秀，上证指数累计上涨了9.3%，位居全球十大股市之首。伴随着大盘向上发力，开放式基金、券商集合理财产品的净值均出现了久违的上涨。整体上看，偏股型券商集合理财产品虽然小有斩获，但是要远远落后于风险相对较高的偏股型基金的阶段表现。开放式基金1月飙涨跌跌不休的A股市场，让2008年的开放式基金经历了从云端坠入谷底的惊天大逆转最终以净值缩水1.34万亿元谢幕，让人唏嘘不已。曾被寄予厚望的开放式基金尤其是股票型基金何时能触底反弹、东山再起？成了不少投资人心中挥之不去的期待。伴随着新年以来大盘的上涨，人们终于看到了希望。银河证券基金研究中心的统计数据显示，1月份，开放式基金迎来了新年的开门红。1月份，股票型、指数型、偏股型和平衡型基金平均上涨6.44%、10.46%、6.29%和4.90%。沪深指数大涨，最为受益的莫过于指数型基金。此外，一些投研实力较强的股票型基金也表现突出，其中股票型基金排名前十位的收益率均不小于12%。农历新年后的第一个交易日，上证指数更是坚强地站在了2000点之上，各偏股型基金净值快速回升。数据显示，截至2月3日，偏股型基金今年来全部取得正收益，涨幅在15%左右的基金随处可见，涨幅超过10%的基金更是比比皆是，连混合偏债型基金都有着3.61%的平均涨幅。根据银河证券的统计，短短一个多月的时间，155只股票型基金(剔除新基金)中，前三甲的涨幅均超过了20%，涨幅在16%~19%的基金有16只，28只基金的涨幅在13%~16%之间，而涨幅在10%~13%之间的基金更是达到了48只，另有60只基金的涨幅在10%以下。券商集合理财产品暂时落后然而，相比于开放式基金尤其是其中的偏股型基金的凌厉涨势，券商集合理财产品则稍微有些“后知后觉”，虽然也出现了不同程度的上涨，但整体而言，显得波澜不惊，明显落后于偏股型基金，且落后于同期大盘。以节前一周的表现为例，上证指数当周涨幅1.85%，深证成指上涨1.41%，股票型开放式基金一周净值收益率平均涨幅1.33%、混合型开放式基金净值平均上涨1.3%，封闭式基金市价平均下跌0.03%；但当周股票型券商集合理财产品平均收益率仅为0.49%，混合型券商集合理财产品平均净值增长率为1.01%，FOF平均净值增长率为0.78%。统计数据显示，截至1月23日，收益最高的偏股型券商集合理财产品当属东方证券旗下的东方红3号，今年来收益为9.83%，国信金理财价值增长和广发理财3号分别以5.96%和4.86%的收益率位列二、三位。
1月收益率排名靠前的偏股型券商集合理财产品还有中金公司旗下的中金股票精选、中金股票策略、中金股票策略二号，累计净值分别为1.8782元、1.0307元和0.763元，成立以来累计净值增长率分别为87.82%、3.07%和-23.7%，今年以来的收益分别为2.14%、1.68%和4.42%；另外，上海证券旗下的理财1号累计净值为0.5376元，今年来的收益为2.8%。 
'''
text=' '.join([w for w in jieba.lcut(text)])
print(text)

vector=tfidf_mdoel.transform([text])
prob=cls_model.predict(vector)
print(prob)
print(le_model.inverse_transform(prob))