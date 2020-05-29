# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:18:07 2019

@author: rkrishnan
"""

import pandas as  pd
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity ,linear_kernel
from numpy import dot
from numpy.linalg import norm
import numpy as np

Doc1=pd.Series(['book book music video video'])
Doc2=pd.Series(['music music video'])
Doc3=pd.Series(['book book video'])


Docs=pd.concat([Doc1,Doc2,Doc3])

Doc1="book book music video video"
Doc2="music music video"
Doc3="book book video"

Docs=[Doc1,Doc2,Doc3]

from sklearn.feature_extraction.text import CountVectorizer

unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True)
vecs_bool=unigram_bool_vectorizer.fit_transform(Docs)
Docs_mx=pd.DataFrame(vecs_bool.toarray(),columns=unigram_bool_vectorizer.get_feature_names())
print(Docs_mx.shape)

X=list(Docs_mx.loc[0])
Y=list(Docs_mx.loc[1])
Z=list(Docs_mx.loc[2])

# cosine similarity using numpy
cos_sim_XY= dot(X,Y)/(norm(X)*norm(Y))
print(cos_sim_XY)
cos_sim_XZ= dot(X,Z)/(norm(X)*norm(Z))
print(cos_sim_XZ)
cos_sim_YZ= dot(Y,Z)/(norm(Y)*norm(Z))
print(cos_sim_YZ)

# cosine similarity using scipy

cos_sim_XY= 1-spatial.distance.cosine(X,Y)
print(cos_sim_XY)
cos_sim_XZ= 1-spatial.distance.cosine(X,Z)
print(cos_sim_XZ)


# cosine similarity using sklearn

cos_sim=cosine_similarity([X,Y,Z])
print(cos_sim)
print()

cos_sim=cosine_similarity([X],[Y,Z])
print(cos_sim)

cos_sim=cosine_similarity([Y],[X,Z])
print(cos_sim)

cos_sim=cosine_similarity(vecs_bool[1],vecs_bool)

cos_sim_sorted_doc_idx=cos_sim.argsort()
# most similar doc byitself
print(Docs[cos_sim_sorted_doc_idx[0][len(Docs)-1]])
# 2nd most similar doc
print(Docs[cos_sim_sorted_doc_idx[0][len(Docs)-2]])
# least similar doc
print(Docs[cos_sim_sorted_doc_idx[0][len(Docs)-3]])


unigram_cnt_vectorizer = CountVectorizer(encoding='latin-1', binary=False)
vecs_cnt=unigram_cnt_vectorizer.fit_transform(Docs)
Docs_mx=pd.DataFrame(vecs_cnt.toarray(),columns=unigram_cnt_vectorizer.get_feature_names())
print(Docs_mx.shape)

X=list(Docs_mx.loc[0])
Y=list(Docs_mx.loc[1])
Z=list(Docs_mx.loc[2])

# cosine similarity using numpy
cos_sim_XY= dot(X,Y)/(norm(X)*norm(Y))
print(cos_sim_XY)
cos_sim_XZ= dot(X,Z)/(norm(X)*norm(Z))
print(cos_sim_XZ)
cos_sim_YZ= dot(Y,Z)/(norm(Y)*norm(Z))
print(cos_sim_YZ)

# cosine similarity using scipy

cos_sim_XY= 1-spatial.distance.cosine(X,Y)
print(cos_sim_XY)
cos_sim_XZ= 1-spatial.distance.cosine(X,Z)
print(cos_sim_XZ)


# cosine similarity using sklearn

cos_sim=cosine_similarity([X,Y,Z])
print(cos_sim)
print()

cos_sim=cosine_similarity([X],[Y,Z])
print(cos_sim)

cos_sim=cosine_similarity([Y],[X,Z])
print(cos_sim)

cos_sim=cosine_similarity(vecs_cnt[1],vecs_cnt)

cos_sim_sorted_doc_idx=cos_sim.argsort()
# most similar doc byitself
print(Docs[cos_sim_sorted_doc_idx[0][len(Docs)-1]])
# 2nd most similar doc
print(Docs[cos_sim_sorted_doc_idx[0][len(Docs)-2]])
# least similar doc
print(Docs[cos_sim_sorted_doc_idx[0][len(Docs)-3]])

