# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 05:13:08 2020

@author: rkrishnan
"""

from nltk import corpus,FreqDist,word_tokenize

files=corpus.gutenberg.fileids()
text=corpus.gutenberg.raw(files[5])
tokens=word_tokenize(text)
words=[w.lower() for w in tokens]

fdist=FreqDist(words)
fdistkeys=list(fdist.keys())
fdistkeys[:50]

fdist['the']
topkeys=fdist.most_common(30)

for pair in topkeys:
    print(pair)
    

numwords=len(words)
topkeysnormalized =[ (word,freq/numwords*100) for (word,freq) in topkeys]

for pair in topkeysnormalized:
    print(pair)