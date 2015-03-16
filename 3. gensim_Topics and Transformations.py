# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 12:31:11 2014

@author: wss
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

# 1、Transformation interface
# 读入字典数据(将语料库中的唯一单词进行编号映射)
dictionary = corpora.Dictionary.load('deerwester.dict')
# 读入Matrix Market格式的语料库文件(词频模型，单词-文档矩阵，即12*9)
corpus = corpora.MmCorpus('deerwester.mm')
print(corpus)
# MmCorpus(9 documents, 12 features, 28 non-zero entries)

# (1)Creating a transformation
# 创建TF-IDF模型
tfidf = models.TfidfModel(corpus)

# (2)Transforming vectors
# 将向量变为TF-IDF
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) 
# [(0, 0.7071067811865476), (1, 0.7071067811865476)]

# 将语料库中的向量全部变为TF-IDF
corpus_tfidf = tfidf[corpus]
# 输出语料库中每篇文档的TF-IDF
for doc in corpus_tfidf:
    print doc

# 初始化LSI模型
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
# 在原始语料库上创建一个双重封装器：bow->tfidf->fold-in-lsi
corpus_lsi = lsi[corpus_tfidf]
# 输出两个隐含维度代表的物理意义 
lsi.print_topics(2)
#topic #0(1.594): -0.703*"trees" + -0.538*"graph" + -0.402*"minors" + -0.187*"survey" + -0.061*"system" + -0.060*"response" + -0.060*"time" + -0.058*"user" + -0.049*"computer" + -0.035*"interface"
#topic #1(1.476): -0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"response" + -0.320*"time" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"
for doc in corpus_lsi:
    print doc
# [(0, -0.066), (1, 0.520)] # "Human machine interface for lab abc computer applications"
# [(0, -0.197), (1, 0.761)] # "A survey of user opinion of computer system response time"
# [(0, -0.090), (1, 0.724)] # "The EPS user interface management system"
# [(0, -0.076), (1, 0.632)] # "System and human system engineering testing of EPS"
# [(0, -0.102), (1, 0.574)] # "Relation of user perceived response time to error measurement"
# [(0, -0.703), (1, -0.161)] # "The generation of random binary unordered trees"
# [(0, -0.877), (1, -0.168)] # "The intersection graph of paths in trees"
# [(0, -0.910), (1, -0.141)] # "Graph minors IV Widths of trees and well quasi ordering"
# [(0, -0.617), (1, 0.054)] # "Graph minors A survey"

# 保存LSI模型和加载LSI模型
# same for tfidf, lda, ...
lsi.save('model.lsi') 
lsi = models.LsiModel.load('model.lsi')

# 2、Available transformations
# (1)Term Frequency * Inverse Document Frequency, Tf-Idf 
model = tfidfmodel.TfidfModel(bow_corpus, normalize=True)
# (2)Latent Semantic Indexing, LSI (or sometimes LSA)
model = lsimodel.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
# (3)Random Projections, RP
model = rpmodel.RpModel(tfidf_corpus, num_topics=500)
# (4)Latent Dirichlet Allocation, LDA 
model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)
# (5)Hierarchical Dirichlet Process, HDP
model = hdpmodel.HdpModel(bow_corpus, id2word=dictionary) 