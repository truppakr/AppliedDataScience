# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:58:45 2019

@author: rkrishnan
"""
import tweepy as tw
import json
import re
import pandas as pd
import os
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain, groupby
import scipy.sparse as sp
from nltk.corpus import stopwords
import math
import warnings 
import collections
from sklearn.feature_extraction.text import TfidfTransformer

warnings.filterwarnings("ignore", category=DeprecationWarning)
#import pycurl
#import StringIO
filename="ai_tweets.txt"
# Start with one tweet:
text = open(filename,encoding='utf-8').read()

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# use parameters to adjust your word cloud, such as 
# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


## use sentiment intesity analyzer to get the sentiment score of alltweets collected

ai_tweets = pd.read_csv(filename,names=['new_tweet'] )
ai_tweets['compound']=0.00
ai_tweets['neg']=0.000
ai_tweets['neu']=0.000
ai_tweets['pos']=0.000
i=-1
sid = SentimentIntensityAnalyzer()
for sentence in ai_tweets['new_tweet']:
     i=1+i
     print(sentence)
     ss = sid.polarity_scores(sentence)
     j=0
     for k in sorted(ss):
         j=j+1
         ai_tweets.iat[i,j]=ss[k]
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()


# tokenization 
#ai_tweets['clean_tweet']=""
## remove special characters
#for row_index in ai_tweets.index:
#    ai_tweets['clean_tweet'][row_index] = ''.join([c for c in ai_tweets['new_tweet'][row_index] if ord(c) < 128])
#    
##eliminating the b' and b" at the begining of the string:
#ai_tweets['new_tweet'] = re.sub(r'^(b\'b\")', '', ai_tweets['new_tweet'])
##deleting the single or double quotation marks at the end of the string:
#ai_tweets['new_tweet'] = re.sub(r'\'\"$', '', ai_tweets['new_tweet'])
##deleting hex
#ai_tweets['new_tweet'] = re.sub(r'\\x[a-f0-9]{2,}', '', ai_tweets['new_tweet'])

# Remove stopwords
# https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
stop = stopwords.words('english')
ai_tweets['new_tweet'] = ai_tweets['new_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
ai_tweets['new_tweet'].head()

# convert to lower case
ai_tweets['new_tweet'] = ai_tweets['new_tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
ai_tweets['new_tweet'].head()

##correct spellings
#ai_tweets['new_tweet'] = ai_tweets['new_tweet'].apply(lambda x: str(TextBlob(x).correct()))
#ai_tweets['new_tweet'].head()


for tweet in ai_tweets['new_tweet']:
    tokens =[t for t in tweet.split()]
    print(tokens)
    freq= nltk.FreqDist(tokens)
#    for key, val in freq.items():
#        print(str(key)+ ":"+ str(val))
#        freq.plot(20, cumulative = False)
        #mytext="Hiking is fun! Hiking with dogs is more fun :)"
#    print(word_tokenize(tokens))

# Tokenization using textblob
from textblob import TextBlob
TextBlob(ai_tweets['new_tweet'][1]).words


# Tokenization using NLTK and converting to a matrix
# https://stackoverflow.com/questions/43962344/tokenization-and-dtmatrix-in-python-with-nltk
word_tokenize_matrix = [word_tokenize(comment) for comment in ai_tweets['new_tweet']]
vocab = set(chain.from_iterable(word_tokenize_matrix))
vocabulary = dict(zip(vocab, range(len(vocab)))) # dictionary of vocabulary to index
sorted_vocabulary = collections.OrderedDict(vocabulary)


freq= nltk.FreqDist(list(vocab))
#for key, val in freq.items():
#    print(str(key)+ ":"+ str(val))
#    freq.plot(100, cumulative = False)


words_index = []
for r, words in enumerate(word_tokenize_matrix):
    for word in sorted(words):
        words_index.append((r, vocabulary.get(word), 1))


#tkn,df = [],[]
#for gid1, d1 in groupby(words_index, lambda x: (x[1])):
#        tkn.append(gid1)
#        df.append(len(list(set(d1))))
#
#DF_count =pd.DataFrame({'tkn': tkn,'df': df})
df = pd.DataFrame(words_index,columns = ["Doc", "Value","Count"])
DF_count=pd.DataFrame(df.groupby(['Value'])['Doc'].nunique())

rows,cols,cnt,row1,tf,tfidf = [],[],[],[],[],[]
for gid, g in groupby(words_index, lambda x: (x[0], x[1])):
        rows.append(gid[0])
        cols.append(gid[1])
        cnt.append(len(list(g)))
        tf.append((cnt[len(cnt)-1]/len(word_tokenize_matrix[gid[0]])))
        tfidf.append(tf[len(tf)-1]*math.log(ai_tweets['new_tweet'].count()/DF_count[DF_count.index== cols[len(cols)-1]]["Doc"].values[0]))
        
X_count = sp.csr_matrix((cnt, (rows, cols)))
V_count = sp.csr_matrix((tf, (rows, cols)))
TFIDF_count =sp.csr_matrix((tfidf, (rows, cols)))

print(X_count)
print(V_count)
print(TFIDF_count)

df_cnt=pd.DataFrame(X_count.todense(),columns=list(sorted_vocabulary))
df_tf=pd.DataFrame(V_count.todense(),columns=list(sorted_vocabulary))
df_tfidf=pd.DataFrame(TFIDF_count.todense(),columns=list(sorted_vocabulary))




# several commonly used vectorizer setting
# The vectorizer can do "fit" and "transform"
# fit is a process to collect unique tokens into the vocabulary
# transform is a process to convert each document to vector based on the vocabulary
# These two processes can be done together using fit_transform(), or used individually: fit() or transform()


#  unigram boolean vectorizer, set minimum document frequency to 1
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=1)
vecs_bool=unigram_bool_vectorizer.fit_transform(ai_tweets['new_tweet'])


#  unigram term frequency vectorizer, set minimum document frequency to 1
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=1)
# fit vocabulary in documents and transform the documents into vectors
vecs = unigram_count_vectorizer.fit_transform(ai_tweets['new_tweet'])


#  unigram and bigram term frequency vectorizer, set minimum document frequency to 1
gram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=1)

#  unigram tfidf vectorizer, set minimum document frequency to 1
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=False, min_df=1)
vecs_tfidf= unigram_tfidf_vectorizer.fit_transform(ai_tweets['new_tweet'])



# Using TfidfTransformer
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(vecs)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': unigram_count_vectorizer.get_feature_names(), 'weight': weights})

# check the content of a document vector
print(vecs.shape)   
print(vecs_tfidf.shape)
print(vecs_bool.shape)

print(vecs[0].toarray())
print(vecs_tfidf[0].toarray())
print(vecs_bool[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_count_vectorizer.vocabulary_))
print(len(unigram_tfidf_vectorizer.vocabulary_))
print(len(unigram_bool_vectorizer.vocabulary_))

# print out the first 10 items in the vocabulary
print(list(unigram_count_vectorizer.vocabulary_.items())[:100])
print(list(unigram_tfidf_vectorizer.vocabulary_.items())[:100])
print(list(unigram_bool_vectorizer.vocabulary_.items())[:100])

# check word index in vocabulary
print(unigram_count_vectorizer.vocabulary_.get('leader'))
print(unigram_tfidf_vectorizer.vocabulary_.get('leader'))
print(unigram_bool_vectorizer.vocabulary_.get('leader'))

#Get feature names
print(unigram_count_vectorizer.get_feature_names())


x=pd.DataFrame(vecs.toarray(),columns=unigram_count_vectorizer.get_feature_names())
y=pd.DataFrame(vecs_tfidf.toarray(),columns=unigram_count_vectorizer.get_feature_names())