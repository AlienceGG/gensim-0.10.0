# -*- coding: utf-8 -*-
"""
Created on Tue Jul 08 07:58:17 2014

@author: wss
"""

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# 1、From Strings to Vectors
import gensim
from gensim import corpora, models, similarities

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
# 对语料库进行分词，移除停顿词['for', 'a', 'of', 'the', 'and', 'to', 'in']
stoplist = set('for a of the and to in'.split()) 
# 推导式可以非常简洁的构造一个新的列表，即[expr for val in collection if condition]
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

# 移除语料库中仅出现一次的单词
# 将列表中的列表连接起来          
all_tokens = __builtins__.sum(texts, [])
# tokens_once存放的是语料库中仅出现一次的单词
tokens_once = set(word for word in set(all_tokens) if all_tokens.cout(word) == 1) 
# texts存放的是语料库，且语料库中的单词满足上述两个条件
texts = [[word for word in text if word not in tokens_once]
          for text in texts]
print(texts)

# 将texts(列表的列表)转变为字典类型，即唯一单词与其唯一编号的映射
dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')
print(dictionary)
# 输出唯一单词与其唯一编号的映射
print(dictionary.token2id)

# 将new_doc表示成为向量空间模型
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print new_vec
# 列表中的元组，第一个表示单词的编号，第二个表示单词出现的个数
# interaction没有出现在字典中，被忽略了
# 其它的10个单词默认出现0次
# [(0, 1), (1, 1)](词频模型)

# 计算语料库中每篇文档的向量空间模型(词频模型)
corpus = [dictionary.doc2bow(text) for text in texts]
# 将corpus存储在磁盘上面，以便后来使用
corpora.MmCorpus('deerwester.mm', corpus)
# 输出语料库中每篇文档的向量空间模型(列表的列表)
print corpus


# 2、Corpus Streaming – One Document at a Time
class MyCorpus(object):
    # 定义自己的迭代器方法
    def __iter__(self):
        for line in open('mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())

# 定义MyCorpus类的对象，此时并没有把语料库加载入内存
corpus_memory_friendly = MyCorpus()
print corpus_memory_friendly
# 输出corpus_memory_friendly对象在内存中的地址
# <__main__.MyCorpus object at 0x03F11D10>

# 一次加载一篇文档，并计算其向量空间模型
for vector in corpus_memory_friendly:
    print(vector)

# construct the dictionary without loading all texts into memory
# 将语料库中的唯一单词进行编号映射
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# 移除停顿词和仅出现一次的单词
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
# 移除停顿词和仅出现一次的词
dictionary.filter_tokens(stop_ids + once_ids)
# 重新将语料库中的唯一单词进行编号映射
dictionary.compactify()
print(dictionary)


# 3、Corpus Formats
# 用Matrix Market格式保存语料库
# 创建两篇文档的语料库，其中一篇文档为空(向量空间模型)
corpus = [[(1, 0.5)], []]
corpora.MmCorpus.serialize('corpus.mm', corpus)

# 分别保存语料库为Joachim’s SVMlight format, Blei’s LDA-C format and GibbsLDA++ format
corpora.SvmLightCorpus.serialize('corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('corpus.lda-c', corpus)
corpora.LowCorpus.serialize('corpus.low', corpus)

# 从Matrix Market文件中加载语料库迭代器
corpus = corpora.MmCorpus('corpus.mm')
# 语料库对象是流，因此你不能直接将其打印
print corpus
# MmCorpus(2 documents, 2 features, 1 non-zero entries)
# (1)查看语料库中的内容
print list(corpus)
# [[(1, 0.5)], []]
# (2)查看语料库的内容
for doc in corpus:    
    print doc
# 结果输出
# [(1, 0.5)]
# []


# 4、Compatibility with NumPy and SciPy
# converting from/to numpy matrices
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
numpy_matrix = gensim.matutils.corpus2dense(corpus)

# from/to scipy.sparse matrices
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)