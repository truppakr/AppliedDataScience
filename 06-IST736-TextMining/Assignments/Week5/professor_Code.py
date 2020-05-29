# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:53:40 2019

@author: rkrishnan
"""


#CountVectorizer_sklearn.py
#Gates
#RE: https://de.dariah.eu/tatom/working_with_text.html
#and
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
## http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-#representation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram

#from sklearn.feature_extraction.text import TfidfTransformer
#import re

##To get the DATA
#Go HERE
#https://de.dariah.eu/tatom/datasets.html#datasets
# Download and save the data used into the files/folders used.

# See: http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-#representation
# For code on IDF and TF_IDF

#vectorizer = CountVectorizer(min_df=1, ngram_range=(1,2), token_pattern=r'\b\w+\b')
#\b means match at exactly the start. r'\bclass\b' matches exactly " class "
#\w+ is one or more chars
# RE: https://docs.python.org/3/howto/regex.html
#corpus = ['This is the first document.',
 #         'This is the second document.',
  #        'And the third one.',
   #       'Is this the first document?']
          
filenames = ['../DATA/Novels_Corpus/Austen_Emma.txt', '../DATA/Novels_Corpus/Austen_Pride.txt', 
             '../DATA/Novels_Corpus/Austen_Sense.txt', '../DATA/Novels_Corpus/CBronte_Jane.txt',
             '../DATA/Novels_Corpus/CBronte_Professor.txt', '../DATA/Novels_Corpus/Dickens_Bleak.txt',
             '../DATA/Novels_Corpus/Dickens_David.txt', '../DATA/Novels_Corpus/Dickens_Hard.txt']

#FILE=open(filenames[0], "r")
#EmmaText=FILE.read()
#FILE.close()
#print(EmmaText[0:100])

vectorizer = CountVectorizer(input='filename')
#Using input='filename' means that fit_transform will
#expect a list of file names
#dtm is document term matrix
dtm = vectorizer.fit_transform(filenames)  # create a sparse matrix
print(type(dtm))
#vocab is a vocabulary list
vocab = vectorizer.get_feature_names()  # change to a list
dtm = dtm.toarray()  # convert to a regular array
print(list(vocab)[500:550])
##Ways to count the word "house" in Emma (file 0 in the list of files)
house_idx = list(vocab).index('house') #index of "house" 
print(house_idx)
print(dtm[0, house_idx]) 
#Counting "house" in Pride
print(dtm[1,house_idx]) 
print(list(vocab)[house_idx]) #his prints "house"
print(dtm[500:550,500:550]) #prints the doc term matrix
#----------
##Create a table of word counts to compare Emma and Pride and Prejudice
columns=["BookName", "house","and","almost"]
MyList=["Emma"]
MyList2=["Pride"]
MyList3=["Sense"]
for someword in ["house","and", "almost"]:
    EmmaWord = (dtm[0, list(vocab).index(someword)])
    MyList.append(EmmaWord)
    PrideWord = (dtm[1, list(vocab).index(someword)])
    MyList2.append(PrideWord)
    SenseWord = (dtm[2, list(vocab).index(someword)])
    MyList3.append(SenseWord)


#print(MyList)
#print(MyList2)

df2=pd.DataFrame([columns, MyList,MyList2, MyList3])   
print(df2)

##Comparing books
#Each row of the document-term 
#matrix (dtm) is a sequence of a novel’s word frequencies
## Get the euclidean dist between Emma and Pride&Prejudice
#Using sklearn

dist = euclidean_distances(dtm)
print(np.round(dist,0))  #The dist between Emma and Pride is 3856

#Measure of distance that takes into account the
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(dtm)
print(np.round(cosdist,3))  #cos dist should be .02

## Visualizing Distances
##An option for visualizing distances is to assign a point in a plane
## to each text such that the distance between points is proportional 
## to the pairwise euclidean or cosine distances.
## This type of visualization is called multidimensional scaling (MDS) 
## in scikit-learn (and R  -  mdscale).

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
## "precomputed" means we will give the dist (as cosine sim)
pos = mds.fit_transform(cosdist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
names=["Austen_Emma", "Austen_Pride", "Austen_Sense", "CBronte_Jane", 
       "CBronte_Professor", "Dickens_Bleak",
       "Dickens_David", "Dickens_Hard"]

for x, y, name in zip(xs, ys, names):
    plt.scatter(x, y)
    plt.text(x, y, name)

plt.show()

##PLotting the relative distances in 3D
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(cosdist)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
    ax.text(x, y, z, s)

ax.set_xlim3d(-.05,.07) #stretch out the x axis
ax.set_ylim3d(-.008,.008) #stretch out the x axis
ax.set_zlim3d(-.05,.05) #stretch out the z axis
plt.show()

## Clustering Texts and Visualizing 
#One method for clustering is Ward’s
#Ward’s method produces a hierarchy of clusterings
# Ward’s method requires  a set of pairwise distance measurements

## The following does not work 
#linkage_matrix = ward(cosdist)
#print(linkage_matrix)
#dendrogram(linkage_matrix, orientation="right", labels=names)
#plt.tight_layout()
#plt.show()

######### Alternative - this works
## Good tutorial http://brandonrose.org/clustering
linkage_matrix = ward(cosdist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=names);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters