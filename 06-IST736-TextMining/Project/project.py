# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:03:04 2019

@author: rkrishnan
"""

import pandas as pd
#from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer 





from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
from sklearn.cluster import KMeans
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from itertools import chain, groupby
import collections
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import matplotlib.pyplot as plt, mpld3
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
stemmer =PorterStemmer()
import os
import math
from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
import re
from PIL import Image
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram
import scipy.sparse as sp
from docopt import docopt
from collections import defaultdict
#class LemmaTokenizer(object):
#    def __init__(self):
#        self.wnl = WordNetLemmatizer()
#    def __call__(self, articles):
#        return [self.wnl.lemmatize(t) for t in word_tokenize(talk)]

# save ted file location in a variable
ted_main_file="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Project\\ted_main.csv"
ted_transcript_file="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Project\\transcripts.csv"

# Read ted files to a dataframe
ted_main = pd.read_csv(ted_main_file) 
ted_transcript = pd.read_csv(ted_transcript_file) 

# Preview the first 5 lines of the loaded data 
ted_main.head()
ted_transcript.head()

# index based on url column
ted_main.index =ted_main.url
ted_transcript.index =ted_transcript.url

# check for duplicates
ted_main_dups=pd.DataFrame(ted_main.duplicated())
ted_main_dups.columns=['dups']
print('Number of duplicates in ted_main dataset: ',sum(ted_main_dups['dups'].apply(lambda x: 1 if x==True else 0)))

ted_transcript_dups=pd.DataFrame(ted_transcript.duplicated())
ted_transcript_dups.columns=['dups']
print('Number of duplicates in ted_transcript dataset: ',sum(ted_transcript_dups['dups'].apply(lambda x: 1 if x==True else 0)))

# There are three duplicates in ted_transcript file and here we are removing those by keeping only
# the 1st occuarance

ted_transcript_clean = ted_transcript.loc[~ted_transcript.index.duplicated(keep='first')]

# now merge both ted_main and ted_transcript into a single dataframe
ted=pd.concat([ted_main,ted_transcript_clean],axis=1,sort=False)
print('Total number of ted talks that had a match in ted main: ', len(ted))

# reset index
ted=ted.reset_index(drop=True)

# Find and remove nas if there are any
ted_clean=ted.dropna()
print('Total number of clean ted talks: ', len(ted_clean))

# drop duplicated columns
ted_clean = ted_clean.loc[:,~ted_clean.columns.duplicated()]

## convert to lower case (not required as we can handle in sklearn)
#ted_clean['transcript'] = ted_clean['transcript'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#ted_clean['transcript'].head()

##remove stopwords (not required here as we can handle in sklearn)
#stop = stopwords.words('english')
#stop.append("the")
#ted_clean['transcript'] = ted_clean['transcript'].apply(lambda x: " ".join(x for x in  x.split() if x not in stop))
#ted_clean['transcript'].head()

# remove numeric tokens
# ted_clean['transcript']=ted_clean['transcript'].apply(lambda x: " ".join(re.sub('^[0-9]+$','',x) for x in x.split())) # didnt work
ted_clean['transcript'] = ted_clean['transcript'].str.replace('\d+', '') 
ted_clean['transcript'].head()

# remove words with 3 or less characters
ted_clean['transcript'] = ted_clean['transcript'].str.replace(r'(\b\w{1,2}\b)', '') # for words

# remove punctuation
ted_clean['transcript'] = ted_clean['transcript'].str.replace('[^\w\s]', ' ') # for punctuation 

# lemmatize tokens
wnl = WordNetLemmatizer()
ted_clean['transcript'] = ted_clean['transcript'].apply(lambda x: " ".join(wnl.lemmatize(x) for x in x.split()))
ted_clean['transcript'].head()

counter = 0
# Now plot wor cloud for each ted talk and see the most common words 
# we have more than 2000 talks and just plotting 1st 10 of them
for index, talk in enumerate(ted_clean['transcript']):
    if counter == 10:
        break
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate((talk))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(ted_clean.loc[index,'url'])
    plt.show()
    counter += 1


# Tokenization using NLTK and converting to a matrix
# https://stackoverflow.com/questions/43962344/tokenization-and-dtmatrix-in-python-with-nltk
word_tokenize_matrix = [word_tokenize(talk) for talk in ted_clean['transcript']]




#  unigram boolean vectorizer, set minimum document frequency to 1
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, 
                                strip_accents = 'unicode', 
                                stop_words = 'english', 
                                lowercase = True, 
                                max_df = 0.95, 
                                min_df = 5
                                )
vecs_bool=unigram_bool_vectorizer.fit_transform(ted_clean['transcript'])


#  unigram term frequency vectorizer, set minimum document frequency to 1
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, 
                                strip_accents = 'unicode', 
                                stop_words = 'english', 
                                lowercase = True,
                                max_df = 0.95, 
                                min_df = 5)
# fit vocabulary in documents and transform the documents into vectors
vecs_cnt = unigram_count_vectorizer.fit_transform(ted_clean['transcript'])

#  unigram tfidf vectorizer, set minimum document frequency to 1
unigram_tf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=False,
                                strip_accents = 'unicode',
                                stop_words = 'english', 
                                lowercase = True, 
                                max_df = 0.95, 
                                min_df = 5)
vecs_tf= unigram_tf_vectorizer.fit_transform(ted_clean['transcript'])

#  unigram tfidf vectorizer, set minimum document frequency to 1
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True,
                                strip_accents = 'unicode',  
                                stop_words = 'english', 
                                lowercase = True, 
                                max_df = 0.95, 
                                min_df = 5)
vecs_tfidf= unigram_tfidf_vectorizer.fit_transform(ted_clean['transcript'])



# check the content of a document vector
print(vecs_bool.shape)
print(vecs_cnt.shape)  
print(vecs_tf.shape) 
print(vecs_tfidf.shape)


# check the size of the constructed vocabulary
print(len(unigram_bool_vectorizer.vocabulary_))
print(len(unigram_count_vectorizer.vocabulary_))
print(len(unigram_tf_vectorizer.vocabulary_))
print(len(unigram_tfidf_vectorizer.vocabulary_))


# print out the first 10 items in the vocabulary
print(list(unigram_bool_vectorizer.vocabulary_.items())[:100])
print(list(unigram_count_vectorizer.vocabulary_.items())[:100])
print(list(unigram_tf_vectorizer.vocabulary_.items())[:100])
print(list(unigram_tfidf_vectorizer.vocabulary_.items())[:100])


# check word index in vocabulary
print(unigram_bool_vectorizer.vocabulary_.get('leader'))
print(unigram_count_vectorizer.vocabulary_.get('leader'))
print(unigram_tf_vectorizer.vocabulary_.get('leader'))
print(unigram_tfidf_vectorizer.vocabulary_.get('leader'))


#Get feature names
print(unigram_bool_vectorizer.get_feature_names())
print(unigram_count_vectorizer.get_feature_names())
print(unigram_tf_vectorizer.get_feature_names())
print(unigram_tfidf_vectorizer.get_feature_names())

df_vecs_bool=pd.DataFrame(vecs_bool.toarray(),columns=unigram_bool_vectorizer.get_feature_names())
df_vecs_cnt=pd.DataFrame(vecs_cnt.toarray(),columns=unigram_count_vectorizer.get_feature_names())
df_vecs_tf=pd.DataFrame(vecs_tf.toarray(),columns=unigram_tf_vectorizer.get_feature_names())
df_vecs_tfidf=pd.DataFrame(vecs_tfidf.toarray(),columns=unigram_tfidf_vectorizer.get_feature_names())

# import ast to convert string representation of a list to list
ted_clean=ted_clean.reset_index(drop=True)
import ast 
df_tags=pd.DataFrame()
df_tags['url']=""
df_tags['tag']=""
df_tags['tags']=""
df_tags['tags'] = df_tags['tags'].astype(object)
i=0
for index,tags_lst in enumerate(ted_clean.tags):
    for tag in ast.literal_eval(tags_lst) :
        df_tags.loc[i,'url']=ted_clean.loc[index,'url']
        df_tags.loc[i,'tag']=tag
        df_tags.loc[i,'tags']=ast.literal_eval(tags_lst)
        i+=1

    
ted_clean_denorm=ted_clean.copy()
ted_clean_denorm.index=ted_clean_denorm['url']
df_tags=df_tags.drop('tags',axis=1)    
df_tags.index=df_tags['url']
df_tags=df_tags.drop('url',axis=1) 

df_tags.join(ted_clean_denorm)
    


#############################################################
########### Clustering Analysis #############################
#############################################################  



## Get the euclidean dist between TED talks
#Using sklearn

dist = euclidean_distances(df_vecs_tf)
print(np.round(dist,0))  #

#Measure of distance that takes into account the
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(df_vecs_tf)
print(np.round(cosdist,3))  #cos dist should be .02


#from sklearn.cluster import KMeans

num_clusters = 10

km = KMeans(n_clusters=num_clusters)

km.fit(cosdist)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


ted_cl = { 'talk': list(ted_clean.index),'cluster': clusters, 'tags': list(ted_clean.tags)}

frame = pd.DataFrame(ted_cl, index = [clusters] , columns = ['talk','cluster', 'tags'])

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)

grouped = frame['talk'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

grouped.mean() #average rank (1 to 100) per cluster

terms = df_vecs_tf.columns

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in ted_clean['transcript']:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_stemmed}, index = totalvocab_tokenized)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d talks:" % i, end='')
    for tag in frame.ix[i]['tags'].values.tolist():
        print(' %s,' % tag, end='')
    print() #add whitespace
    print() #add whitespace
    
print()
print()

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

#set up colors per clusters using a dict
cluster_colors = { 0: '#543005',1: '#8c510a', 2: '#bf812d', 3: '#dfc27d', 4: '#f6e8c3', 5: '#f5f5f5', 6: '#c7eae5', 
                  7: '#80cdc1', 8: '#35978f', 9: '#01665e'}

#set up cluster names using a dict
cluster_names = {0: 'Cluster 1', 
                 1: 'Cluster 2',
                 2: 'Cluster 3',
                 3: 'Cluster 4',
                 4: 'Cluster 5',
                 5: 'Cluster 6',
                 6: 'Cluster 7',
                 7: 'Cluster 8',
                 8: 'Cluster 9',
                 9: 'Cluster 10'}

for x, y, name in zip(xs, ys, df_vecs_tf.index):
    plt.scatter(x, y)
    plt.text(x, y, name)

plt.show()


#some ipython magic to show the matplotlib plots inline
#%matplotlib inline 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#create data frame that has the result of the MDS plus the cluster numbers and tags
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, ted_tags=(ted_clean['tags']))) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['ted_tags'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

##PLotting the relative distances in 3D
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(cosdist)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], df_vecs_cnt.index):
    ax.text(x, y, z, s)

ax.set_xlim3d(-.5,.5) #stretch out the x axis
ax.set_ylim3d(-.5,.5) #stretch out the x axis
ax.set_zlim3d(-.5,.5) #stretch out the z axis
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

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, ted=(ted_clean['tags']))) 
#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot 
fig, ax = plt.subplots(figsize=(18,18)) #set plot size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.ted]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    
ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

mpld3.enable_notebook()
#uncomment the below to export to html
html = mpld3.fig_to_html(fig)

f = open('cluster.html','w')

f.write(html)
f.close()

######### Alternative - this works
## Good tutorial http://brandonrose.org/clustering
#from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix = ward(cosdist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(10, 10)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=ted_clean.tags);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

mpld3.plugins.connect(fig, tooltip, TopToolbar())   
#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

html = mpld3.fig_to_html(fig)

f = open('dendogram.html','w')

f.write(html)
f.close()

