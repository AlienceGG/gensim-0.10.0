# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 05:43:28 2014

@author: wss
"""

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
# gensim is a topic modelling for humans

# a small corpus of nine documents and twelve features
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]
# initialize a transformation
tfidf = models.TfidfModel(corpus)
# vec相当于一篇文档的VSM
vec = [(0, 1), (4, 1)]
# 输出上文的TF-IDF值
print tfidf[vec]
# 把语料库通过TF-IDF模型，全部变为词频-逆向文件频率(单词-文档矩阵，即12*9)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
# query the similarity of our query vector vec against every document in the corpus
sims = index[tfidf[vec]]
# 在同时需要用到index和value值的时候可以用到enumerate，参数为可遍历的变量，返回enumerate类
print list(enumerate(sims))

# 说明：本部分的重点是TF-IDF模型