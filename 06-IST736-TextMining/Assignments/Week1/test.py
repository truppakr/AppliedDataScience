# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:24:26 2019

@author: rkrishnan
"""


#tweets = pd.read_csv("saved_tweets.csv",encoding = 'unicode_escape')  
#tweets.head()  

import pandas as pd  
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

text ="To be or not to be"
tokens =[t for t in text.split()]
print(tokens)

freq= nltk.FreqDist(tokens)

for key, val in freq.items():
    print(str(key)+ ":"+ str(val))

freq.plot(20, cumulative = False)

mytext="Hiking is fun! Hiking with dogs is more fun :)"
    
print(word_tokenize(mytext))


from nltk.corpus import brown
brown.words()


import re
str = 'This is a tweet with a url: http://t.co/0DlGChTBIx'
m = re.sub(r':.*$', ":", str)