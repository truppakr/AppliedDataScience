# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 03:05:06 2019

@author: rkrishnan
"""

# read in the training data

# the data set includes four columns: PhraseId, SentenceId, Phrase, Sentiment
# In this data set a sentence is further split into phrases 
# in order to build a sentiment classification model
# that can not only predict sentiment of sentences but also shorter phrases

# A data example:
# PhraseId SentenceId Phrase Sentiment
# 1 1 A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .1

# the Phrase column includes the training examples
# the Sentiment column includes the training labels
# "0" for very negative
# "1" for negative
# "2" for neutral
# "3" for positive
# "4" for very positive

import numpy as np
import pandas as p
from ast import literal_eval
import re

train=p.read_csv("C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week6\\kaggle-sentiment\\kaggle-sentiment\\train.tsv", delimiter='\t')
y=train['Sentiment'].values
X=train['Phrase'].values
X1=p.DataFrame()
X1['Phrase']=train['Phrase'].apply(lambda x: " ".join(re.sub('n\'t',' not',x) for x in x.split()))
X1=X1['Phrase'].values


