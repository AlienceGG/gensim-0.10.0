# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 13:54:11 2014

@author: wss
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
# 1、Similarity interface
dictionary = corpora.Dictionary.load('deerwester.dict')
corpus = corpora.MmCorpus('deerwester.mm') 
print(corpus)
# MmCorpus(9 documents, 12 features, 28 non-zero entries)

# 使用上面的语料库来定义一个2维的LSI空间
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# 定义查询文档
doc = "Human computer interaction"
# 将查询文档变为向量空间模型(词袋模型；词频模型)
vec_bow = dictionary.doc2bow(doc.lower().split())
# convert the query to LSI space
vec_lsi = lsi[vec_bow] 
print(vec_lsi)
# [(0, -0.461821), (1, 0.070028)]


# Initializing query structures
# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi[corpus])

# 保存和打开索引函数
index.save('deerwester.index')
index = similarities.MatrixSimilarity.load('deerwester.index')


# 3、Performing queries
# 计算查询文档与其它9个文档的相似度
sims = index[vec_lsi] 
# print (document_number, document_similarity) 2-tuples
print(list(enumerate(sims))) 
# [(0, 0.99809301), (1, 0.93748635), (2, 0.99844527), (3, 0.9865886), (4, 0.90755945),
# (5, -0.12416792), (6, -0.1063926), (7, -0.098794639), (8, 0.05004178)]
# 计算查询文档与其它9个文档的相似度，并按降序排序
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)
# [(2, 0.99844527), # The EPS user interface management system
# (0, 0.99809301), # Human machine interface for lab abc computer applications
# (3, 0.9865886), # System and human system engineering testing of EPS
# (1, 0.93748635), # A survey of user opinion of computer system response time
# (4, 0.90755945), # Relation of user perceived response time to error measurement
# (8, 0.050041795), # Graph minors A survey
# (7, -0.098794639), # Graph minors IV Widths of trees and well quasi ordering
# (6, -0.1063926), # The intersection graph of paths in trees
# (5, -0.12416792)] # The generation of random binary unordered trees


# 如果处理大量的数据，可以采用分布式，比如Pyro