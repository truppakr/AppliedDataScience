# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:09:27 2020

@author: rkrishnan
"""

import nltk
from nltk import FreqDist
import re
from nltk.collocations import *

print (nltk.corpus.gutenberg.fileids())
file14 = nltk.corpus.gutenberg.fileids()[14]
miltontext = nltk.corpus.gutenberg.raw(file14)
miltontokens = nltk.word_tokenize(miltontext) 
miltonwords = [w.lower( ) for w in miltontokens]
# show the number of words and print the first 110 words
print(len(miltonwords))
print(miltonwords[ :110])


ndist = FreqDist(miltonwords)
nitems = ndist.most_common(30)
for item in nitems:
    print (item[0], '\t', item[1])

miltonwords2 = nltk.corpus.gutenberg.words('milton-paradise.txt')
miltonwords2lowercase = [w.lower() for w in miltonwords2]


len(miltonwords)
len(miltonwords2lowercase)

print(miltonwords[:160])
print(miltonwords2lowercase[:160])

# this regular expression pattern matches any word that contains all non-alphabetical
#   lower-case characters [^a-z]+
# the beginning ^ and ending $ require the match to begin and end on a word boundary 
pattern = re.compile('^[^a-z]+$')

# function that takes a word and returns true if it consists only
#   of non-alphabetic characters

def alpha_filter(w):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False

def verb_forms_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^\w+(?:ing|ed)$')
  if (pattern.match(w)):
    return False
  else:
    return True

alphamiltonwords = [w for w in miltonwords if not alpha_filter(w)]
print(len(alphamiltonwords))
print(alphamiltonwords[:100])

alphamiltonwords = [w for w in alphamiltonwords if not verb_forms_filter(w)]
print(len(alphamiltonwords))
print(alphamiltonwords[:100])

nltkstopwords = nltk.corpus.stopwords.words('english')
print(len(nltkstopwords))
print(nltkstopwords)

print(miltonwords[:100])
print(miltonwords[15300:15310])

morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve"]


stopwords = nltkstopwords + morestopwords
print(len(stopwords))
print(stopwords)


stoppedmiltonwords = [w for w in alphamiltonwords if not w in stopwords]
print(len(stoppedmiltonwords))

miltondist = FreqDist(stoppedmiltonwords)
miltonitems = miltondist.most_common(30)
for item in miltonitems:
  print(item)


miltonbigrams = list(nltk.bigrams(miltonwords))

print(miltonbigrams[:20])



# Next, for convenience, we define a variable for the bigram measures.
bigram_measures = nltk.collocations.BigramAssocMeasures()


finder = BigramCollocationFinder.from_words(miltonwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)

print(type(scored))
first = scored[0]
print(type(first), first)

for bscore in scored[:30]:
    print (bscore)


finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:30]:
    print (bscore)


finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:30]:
    print (bscore)


finder2 = BigramCollocationFinder.from_words(miltonwords)
finder2.apply_freq_filter(2)
scored = finder2.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:20]:
    print (bscore)


finder2.apply_ngram_filter(lambda w1, w2: len(w1) < 2)
scored = finder2.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:20]:
    print (bscore)

#Mutual Information and other scorers
#
#Recall that Mutual Information is a score introduced in the paper by Church and Hanks, where they defined it as an Association Ratio. Note that technically the original information theoretic definition of mutual information allows the two words to be in either order, but that the association ratio defined by Church and Hanks requires the words to be in order from left to right wherever they appear in the window
#
#In NLTK, the mutual information score is given by a function for Pointwise Mutual Information, where this is the version without the window.


finder3 = BigramCollocationFinder.from_words(miltonwords)
scored = finder3.score_ngrams(bigram_measures.pmi)
for bscore in scored[:30]:
    print (bscore)


finder3.apply_freq_filter(5)
scored = finder3.score_ngrams(bigram_measures.pmi)
for bscore in scored[:30]:
    print (bscore)
