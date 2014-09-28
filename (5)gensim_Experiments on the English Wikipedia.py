# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 14:43:01 2014

@author: wss
"""

import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 1、Preparing the corpus
# 直接将维基百科语料库转换为TF-IDF向量的形式
# python -m gensim.scripts.make_wiki


# 2、Latent Semantic Analysis
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# 加载语料库的迭代器
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# 输出mm
print(mm)
# MmCorpus(3931787 documents, 100000 features, 756379027 non-zero entries)

# 创建维基百科语料库的LSI模型
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)
# 输出前10主题
lsi.print_topics(10)
#topic #0(332.762): 0.425*"utc" + 0.299*"talk" + 0.293*"page" + 0.226*"article" + 0.224*"delete" + 0.216*"discussion" + 0.205*"deletion" + 0.198*"should" + 0.146*"debate" + 0.132*"be"
#topic #1(201.852): 0.282*"link" + 0.209*"he" + 0.145*"com" + 0.139*"his" + -0.137*"page" + -0.118*"delete" + 0.114*"blacklist" + -0.108*"deletion" + -0.105*"discussion" + 0.100*"diff"
#topic #2(191.991): -0.565*"link" + -0.241*"com" + -0.238*"blacklist" + -0.202*"diff" + -0.193*"additions" + -0.182*"users" + -0.158*"coibot" + -0.136*"user" + 0.133*"he" + -0.130*"resolves"
#topic #3(141.284): -0.476*"image" + -0.255*"copyright" + -0.245*"fair" + -0.225*"use" + -0.173*"album" + -0.163*"cover" + -0.155*"resolution" + -0.141*"licensing" + 0.137*"he" + -0.121*"copies"
#topic #4(130.909): 0.264*"population" + 0.246*"age" + 0.243*"median" + 0.213*"income" + 0.195*"census" + -0.189*"he" + 0.184*"households" + 0.175*"were" + 0.167*"females" + 0.166*"males"
#topic #5(120.397): 0.304*"diff" + 0.278*"utc" + 0.213*"you" + -0.171*"additions" + 0.165*"talk" + -0.159*"image" + 0.159*"undo" + 0.155*"www" + -0.152*"page" + 0.148*"contribs"
#topic #6(115.414): -0.362*"diff" + -0.203*"www" + 0.197*"you" + -0.180*"undo" + -0.180*"kategori" + 0.164*"users" + 0.157*"additions" + -0.150*"contribs" + -0.139*"he" + -0.136*"image"
#topic #7(111.440): 0.429*"kategori" + 0.276*"categoria" + 0.251*"category" + 0.207*"kategorija" + 0.198*"kategorie" + -0.188*"diff" + 0.163*"категория" + 0.153*"categoría" + 0.139*"kategoria" + 0.133*"categorie"
#topic #8(109.907): 0.385*"album" + 0.224*"song" + 0.209*"chart" + 0.204*"band" + 0.169*"released" + 0.151*"music" + 0.142*"diff" + 0.141*"vocals" + 0.138*"she" + 0.132*"guitar"
#topic #9(102.599): -0.237*"league" + -0.214*"he" + -0.180*"season" + -0.174*"football" + -0.166*"team" + 0.159*"station" + -0.137*"played" + -0.131*"cup" + 0.131*"she" + -0.128*"utc"


# 3、Latent Dirichlet Allocation
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# 加载语料库迭代器
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# 输出mm
print(mm)
# MmCorpus(3931787 documents, 100000 features, 756379027 non-zero entries)

# 创建LDA模型
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
# using serial LDA version on this node
# running online LDA training, 100 topics, 1 passes over the supplied corpus of 3931787 documents, updating model once every 10000 documents
# ...

# print the most contributing words for 20 randomly selected topics
lda.print_topics(20)
#topic #0: 0.009*river + 0.008*lake + 0.006*island + 0.005*mountain + 0.004*area + 0.004*park + 0.004*antarctic + 0.004*south + 0.004*mountains + 0.004*dam
#topic #1: 0.026*relay + 0.026*athletics + 0.025*metres + 0.023*freestyle + 0.022*hurdles + 0.020*ret + 0.017*divisão + 0.017*athletes + 0.016*bundesliga + 0.014*medals
#topic #2: 0.002*were + 0.002*he + 0.002*court + 0.002*his + 0.002*had + 0.002*law + 0.002*government + 0.002*police + 0.002*patrolling + 0.002*their
#topic #3: 0.040*courcelles + 0.035*centimeters + 0.023*mattythewhite + 0.021*wine + 0.019*stamps + 0.018*oko + 0.017*perennial + 0.014*stubs + 0.012*ovate + 0.011*greyish
#topic #4: 0.039*al + 0.029*sysop + 0.019*iran + 0.015*pakistan + 0.014*ali + 0.013*arab + 0.010*islamic + 0.010*arabic + 0.010*saudi + 0.010*muhammad
#topic #5: 0.020*copyrighted + 0.020*northamerica + 0.014*uncopyrighted + 0.007*rihanna + 0.005*cloudz + 0.005*knowles + 0.004*gaga + 0.004*zombie + 0.004*wigan + 0.003*maccabi
#topic #6: 0.061*israel + 0.056*israeli + 0.030*sockpuppet + 0.025*jerusalem + 0.025*tel + 0.023*aviv + 0.022*palestinian + 0.019*ifk + 0.016*palestine + 0.014*hebrew
#topic #7: 0.015*melbourne + 0.014*rovers + 0.013*vfl + 0.012*australian + 0.012*wanderers + 0.011*afl + 0.008*dinamo + 0.008*queensland + 0.008*tracklist + 0.008*brisbane
#topic #8: 0.011*film + 0.007*her + 0.007*she + 0.004*he + 0.004*series + 0.004*his + 0.004*episode + 0.003*films + 0.003*television + 0.003*best
#topic #9: 0.019*wrestling + 0.013*château + 0.013*ligue + 0.012*discus + 0.012*estonian + 0.009*uci + 0.008*hockeyarchives + 0.008*wwe + 0.008*estonia + 0.007*reign
#topic #10: 0.078*edits + 0.059*notability + 0.035*archived + 0.025*clearer + 0.022*speedy + 0.021*deleted + 0.016*hook + 0.015*checkuser + 0.014*ron + 0.011*nominator
#topic #11: 0.013*admins + 0.009*acid + 0.009*molniya + 0.009*chemical + 0.007*ch + 0.007*chemistry + 0.007*compound + 0.007*anemone + 0.006*mg + 0.006*reaction
#topic #12: 0.018*india + 0.013*indian + 0.010*tamil + 0.009*singh + 0.008*film + 0.008*temple + 0.006*kumar + 0.006*hindi + 0.006*delhi + 0.005*bengal
#topic #13: 0.047*bwebs + 0.024*malta + 0.020*hobart + 0.019*basa + 0.019*columella + 0.019*huon + 0.018*tasmania + 0.016*popups + 0.014*tasmanian + 0.014*modèle
#topic #14: 0.014*jewish + 0.011*rabbi + 0.008*bgwhite + 0.008*lebanese + 0.007*lebanon + 0.006*homs + 0.005*beirut + 0.004*jews + 0.004*hebrew + 0.004*caligari
#topic #15: 0.025*german + 0.020*der + 0.017*von + 0.015*und + 0.014*berlin + 0.012*germany + 0.012*die + 0.010*des + 0.008*kategorie + 0.007*cross
#topic #16: 0.003*can + 0.003*system + 0.003*power + 0.003*are + 0.003*energy + 0.002*data + 0.002*be + 0.002*used + 0.002*or + 0.002*using
#topic #17: 0.049*indonesia + 0.042*indonesian + 0.031*malaysia + 0.024*singapore + 0.022*greek + 0.021*jakarta + 0.016*greece + 0.015*dord + 0.014*athens + 0.011*malaysian
#topic #18: 0.031*stakes + 0.029*webs + 0.018*futsal + 0.014*whitish + 0.013*hyun + 0.012*thoroughbred + 0.012*dnf + 0.012*jockey + 0.011*medalists + 0.011*racehorse
#topic #19: 0.119*oblast + 0.034*uploaded + 0.034*uploads + 0.033*nordland + 0.025*selsoviet + 0.023*raion + 0.022*krai + 0.018*okrug + 0.015*hålogaland + 0.015*russiae + 0.020*manga + 0.017*dragon + 0.012*theme + 0.011*dvd + 0.011*super + 0.011*hunter + 0.009*ash + 0.009*dream + 0.009*angel

# run batch LDA (not online), train LdaModel
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)
doc_lda = lda[doc_bow]