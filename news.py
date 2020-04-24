import requests
from bs4 import BeautifulSoup
import jieba
import jieba.posseg as pseg
import re
from wordcloud import WordCloud
from textrank4zh import TextRank4Keyword, TextRank4Sentence

url = 'http://baijiahao.baidu.com/s?id=1664798548558080117'
html = requests.get(url,timeout=10)
content = html.content
# print(content)

# 创建BS对象
soup = BeautifulSoup(content,'html.parser',from_encoding='utf-8')
text = soup.get_text()
# print(text)

words = pseg.lcut(text)
# 新闻中出现的人物和地名
news_person = {word for word,flag in words if flag == 'nr'}
news_place = {word for word,flag in words if flag == 'ns'}
print('新闻中出现的人物有:',news_person)
print('新闻中出现的地名有:',news_place)

# 提取中文,去掉非中文
text = re.sub('[^\u4e00-\u9fa5。，！：；、]','',text)
# print(text)

# 去掉停用词
def remove_stop_words(f):
    for stop_word in ['']:
        f = f.replace(stop_word,'')
    return f
def creat_word_cloud(f):
    f = remove_stop_words(f)
    text = " ".join(jieba.lcut(f))
    wc = WordCloud(
        max_words=100,
        width=2000,
        height=1200,
        font_path='msyh.ttf'
    )
    wordcloud = wc.generate(text)
    wordcloud.to_file("wordcloud.jpg")

# 生成词云
creat_word_cloud(text)

# 关键词提取
tr4w = TextRank4Keyword()
tr4w.analyze(text=text, lower=True, window=3)
print('关键词：')
for item in tr4w.get_keywords(10, word_min_len=2):
    print(item.word, item.weight)

# 生成摘要
tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source='all_filters')
print('摘要：')
for item in tr4s.get_key_sentences(num=2):
    print(item.index, item.weight, item.sentence)