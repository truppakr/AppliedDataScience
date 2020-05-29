​# -*- coding: utf-8 -*-
"""


@author: profa
"""

###################################################
##
## LDA for Topic Modeling
##
###################################################

## DATA USED IS FROM KAGGLE
##
## https://www.kaggle.com/therohk/million-headlines/version/7

## Tutorial and code taken from 
## https://towardsdatascience.com/
## topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

## Other good tutorials
## https://nlpforhackers.io/topic-modeling/
# https://www.kaggle.com/meiyizi/spooky-nlp-and-topic-modelling-tutorial

import pandas as pd

#data = pd.read_csv('DATA/abcnews_date_text_Kaggle.csv', error_bad_lines=False);
data_small=pd.read_csv('DATA/abcnews_date_text_Kaggle_subset100.csv', error_bad_lines=False);
print(data_small.head())
## headline_text is the column name for the headline in the dataset
#data_text = data[['headline_text']]
data_text_small = data_small[['headline_text']]
print(data_text_small)

#data_text['index'] = data_text.index
data_text_small['index'] = data_text_small.index
#print(data_text_small.index)
#print(data_text_small['index'])

#documents = data_text
documents = data_text_small
print(documents)

print("The length of the file - or number of docs is", len(documents))
print(documents[:5])

###################################################
###
### Data Prep and Pre-processing
###
###################################################

import gensim
## IMPORTANT - you must install gensim first ##
## conda install -c anaconda gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')
from nltk import PorterStemmer
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()

from nltk.tokenize import word_tokenize 
from nltk.stem.porter import *

#NOTES
##### Installing gensim caused my Spyder IDE no fail and no re-open
## I used two things and did a restart
## 1) in cmd (if PC)  psyder --reset
## 2) in cmd (if PC) conda upgrade qt

######################################
## function to perform lemmatize and stem preprocessing
############################################################
## Function 1
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#Select a document to preview after preprocessing
doc_sample = documents[documents['index'] == 50].values[0][0]
print(doc_sample)
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))



## Preprocess the headline text, saving the results as ‘processed_docs’
processed_docs = documents['headline_text'].map(preprocess)
print(processed_docs[:10])


## Create a dictionary from ‘processed_docs’ containing the 
## number of times a word appears in the training set.

dictionary = gensim.corpora.Dictionary(processed_docs)

## Take a look ...you can set count to any number of items to see
## break will stop the loop when count gets to your determined value
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 5:
        break
    
#print(processed_docs)   
## Filter out tokens that appear in
## - - less than 15 documents (absolute number) or
## - - more than 0.5 documents (fraction of total corpus size, not absolute number).
## - - after the above two steps, keep only the first 100000 most frequent tokens
 ############## NOTE - this line of code did not work with my small sample
## as it created blank lists.....       
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

for doc in processed_docs:
    print(doc)

print(dictionary)
#######################
## For each document we create a dictionary reporting how many
##words and how many times those words appear. Save this to ‘bow_corpus’
##############################################################################
#### bow: Bag Of Words
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[3:5])


#################################################################
### TF-IDF
#################################################################
##Create tf-idf model object using models.TfidfModel on ‘bow_corpus’ 
## and save it to ‘tfidf’, then apply transformation to the entire 
## corpus and call it ‘corpus_tfidf’. Finally we preview TF-IDF 
## scores for our first document.

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
## pprint is pretty print
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    ## the break will stop it after the first doc
    break



#############################################################
### Running LDA using Bag of Words
#################################################################
    
#lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary, passes=1)
    
################################################################
## sklearn
###################################################################3
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
 
NUM_TOPICS = 3

filenames = ['../DATA/Austen_Emma.txt', '../DATA/Austen_Pride.txt', 
             '../DATA/Austen_Sense.txt', '../DATA/CBronte_Jane.txt',
             '../DATA/CBronte_Professor.txt', '../DATA/Dickens_Bleak.txt',
             '../DATA/Dickens_David.txt', '../DATA/Dickens_Hard.txt']

MyVectLDA=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
data_vectorized = MyVectLDA.fit_transform(filenames)
ColumnNamesLDA=MyVectLDA.get_feature_names()
CorpusDF_LDA=pd.DataFrame(data_vectorized.toarray(),columns=ColumnNamesLDA)
CorpusDF_LDA = CorpusDF_LDA[CorpusDF_LDA.columns.drop(list(CorpusDF_LDA.filter(regex='\d+')))]
CorpusDF_LDA = CorpusDF_LDA[CorpusDF_LDA.columns.drop(list(CorpusDF_LDA.filter(regex='\_+')))]
print(CorpusDF_LDA)


lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)
lda_Z_DF = lda_model.fit_transform(CorpusDF_LDA)
print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components=NUM_TOPICS)
nmf_Z = nmf_model.fit_transform(CorpusDF_LDA)
print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
lsi_Z = lsi_model.fit_transform(CorpusDF_LDA)
print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
 
 
# Let's see how the first document in the corpus looks like in
## different topic spaces
print(lda_Z_DF[0])
print(nmf_Z[0])
print(lsi_Z[0])

## implement a print function 
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
 
print("LDA Model:")
print_topics(lda_model, MyVectLDA)
#print("=" * 20)
 
print("NMF Model:")
print_topics(nmf_model, MyVectLDA)
#print("=" * 20)
 
print("LSI Model:")
print_topics(lsi_model, MyVectLDA)
#print("=" * 20)



#########################################
## Try sklean LDA with DOG and HIKE data
##########################################
import os
all_file_names = []

path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\Dog_Hike"
#print("calling os...")
#print(os.listdir(path))
FileNameList=os.listdir(path)
#print(FileNameList)
ListOfCompleteFiles=[]
for name in os.listdir(path):
    print(path+ "\\" + name)
    next=path+ "\\" + name
    ListOfCompleteFiles.append(next)
#print("DONE...")
print("full list...")
print(ListOfCompleteFiles)


MyVectLDA_DH=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
Vect_DH = MyVectLDA_DH.fit_transform(ListOfCompleteFiles)
ColumnNamesLDA_DH=MyVectLDA_DH.get_feature_names()
CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
print(CorpusDF_DH)


lda_model_DH = LatentDirichletAllocation(n_components=2, max_iter=10, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(Vect_DH)

print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in Dog and Hike data...")
print(LDA_DH_Model[0])
print("Seventh Doc in DOg Hike...")
print(LDA_DH_Model[6])

## Print LDA using print function from above
print("LDA Dog and Hike Model:")
print_topics(lda_model_DH, MyVectLDA_DH)


####################################################
##
## VISUALIZATION
##
####################################################
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model_DH, Vect_DH, MyVectLDA_DH, mds='tsne')
pyLDAvis.show(panel)