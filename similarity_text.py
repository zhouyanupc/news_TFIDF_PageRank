import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载停用词
with open(r'chinese_stopwords.txt','r',encoding='utf-8') as f:
    stopwords = [line[:-1] for line in f.readlines()]
# print(stopwords)
# 数据加载
path = r'sqlResult.csv'
news = pd.read_csv(path,encoding='gb18030')
print(news.shape)
print(news.head())
# 处理缺失值
print(news[news.content.isna()].head())
news = news.dropna(subset=['content'])
print(news.shape)

# 分词
def split_text(text):
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text2 = jieba.cut(text.strip())
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result

# text = news.iloc[0].content
# print(text)
# print(split_text(text))

import pickle, os
if not os.path.exists('corpus.pkl'):
    corpus = list(map(split_text,[str(i) for i in news.content]))
    print(corpus[0])
    print(len(corpus))
    print(corpus[1])
    with open('corpus.pkl', 'wb') as f:
        pickle.dump(corpus,f)
else:
    with open('corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)

# 计算corpus的TF-IDF矩阵
countvectorizer = CountVectorizer(encoding='gb18030',min_df=0.05)
tfidftransformer = TfidfTransformer()
countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidftransformer.fit_transform(countvector)
print(tfidf.shape)

# 标记是否是新华社的新闻
lable = list(map(lambda source:1 if '新华' in str(source) else 0 , news.source))
# 数据切分
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
x_train, x_test, y_train, y_test = train_test_split(tfidf.toarray(), lable, test_size=0.3,random_state=2)
clf = MultinomialNB()
clf.fit(x_train,y_train)
prediction = clf.predict(tfidf.toarray())
lables = np.array(lable)

compare_news_index = pd.DataFrame({'prediction':prediction,'lables':lables})
# 计算所有可疑文章的index
copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['lables'] == 0)].index

xinhua_news_index = compare_news_index[(compare_news_index['lables'] == 1)].index
print('可疑文章数:', len(copy_news_index))

# 聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
scaled_array = normalizer.fit_transform(tfidf.toarray())

if not os.path.exists('k_labels.pkl'):
    kmeans = KMeans(n_clusters=25)
    k_labels = kmeans.fit_transform(scaled_array)
    with open('k_labels.pkl', 'wb') as f:
        pickle.dump(k_labels,f)
else:
    with open('k_labels.pkl', 'rb') as f:
        k_labels = pickle.load(f)


# 创建id_class

if not os.path.exists('id_class.pkl'):
    id_class = {index:class_ for index, class_ in enumerate(k_labels)}
    with open('id_class.pkl', 'wb') as f:
        pickle.dump(id_class,f)
else:
    with open('id_class.pkl', 'rb') as f:
        id_class = pickle.load(f)

from collections import defaultdict
if not os.path.exists('class_id.pkl'):
    class_id = defaultdict(set)
    for index, class_ in id_class.items():
        if index in xinhua_news_index.tolist():
            class_id[tuple(class_.tolist())].add(index)
    with open('class_id.pkl', 'wb') as f:
        pickle.dump(class_id,f)
else:
    with open('class_id.pkl', 'rb') as f:
        class_id = pickle.load(f)
# print(class_id)
# 相似文本
def find_similar_text(cpindex, top=10):
    dist_dict = {i:cosine_similarity(tfidf[cpindex],tfidf[i]) for i in class_id[tuple(id_class[cpindex].tolist())]}
    return sorted(dist_dict.items(),key=lambda x:x[1][0], reverse=True)[:top]

cpindex = 78657
similar_list = find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭:\n', news.iloc[cpindex].content)

similar_2 = similar_list[0][0]
print('相似原文\n', news.iloc[similar_2].content)

import editdistance
print('编辑距离:',editdistance.eval(corpus[cpindex], corpus[similar_2]))
