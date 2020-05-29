# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:58:45 2019

@author: rkrishnan
"""

####################################################
###
### This code is an example of 
### tokenization
### vectorization
### Dealing with a corpus
###  k means
### Dealing with a labeled CSV
### DIstance measures
### Frequency
### Normalization
### Formats

####################################################
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
from nltk.stem import Porter



ps =PorterStemmer()
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

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

#import pycurl
#import StringIO
filename="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week5\\ai_tweets_tw.txt"
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


tokens =[t for t in text.split()]
print(tokens)
#freq= nltk.FreqDist(tokens)
#for key, val in freq.items():
#    #print(str(key)+ ":"+ str(val))
#    freq.plot(1000, cumulative = False)
#print(word_tokenize(tokens))


## use sentiment intesity analyzer to get the sentiment score of alltweets collected

ai_tweets = pd.read_csv(filename,names=['new_tweet'] )


# Tokenization using textblob
from textblob import TextBlob
TextBlob(ai_tweets['new_tweet'][1]).words
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

# remove artificial intellegence keyword 
ai_tweets['new_tweet'] = [item.replace('artificial intelligence', 'AI') for item in ai_tweets.new_tweet]
ai_tweets['new_tweet'].head()

##correct spellings
#ai_tweets['new_tweet'] = ai_tweets['new_tweet'].apply(lambda x: str(TextBlob(x).correct()))
#ai_tweets['new_tweet'].head()

#mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/v/o/v/j/p/S/70-hi.png', stream=True).raw))
mask = np.array(Image.open('C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week5\\AI4.png'))
#mask_pos = np.array(Image.open(requests.get('http://www.clker.com/cliparts/v/c/U/J/W/e/speedmeter-hi.png', stream=True).raw))
mask_pos = np.array(Image.open('C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week4\\positive.JPG'))
mask_neg = np.array(Image.open('C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week4\\negative.JPG'))
mask_neu = np.array(Image.open('C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week4\\neutral.JPG'))


image_colors = ImageColorGenerator(mask)


#fig, axes = plt.subplots(1, 3)
#axes[0].imshow(wc, interpolation="bilinear")
## recolor wordcloud and show
## we could also give color_func=image_colors directly in the constructor
#axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
#axes[2].imshow(mask, cmap=plt.cm.gray, interpolation="bilinear")
#for ax in axes:
#    ax.set_axis_off()
#plt.show()

## use sentiment intesity analyzer to get the sentiment score of alltweets collected


ai_tweets['compound']=0.00
ai_tweets['neg']=0.000
ai_tweets['neu']=0.000
ai_tweets['pos']=0.000
sid = SentimentIntensityAnalyzer()
for index,sentence in enumerate(ai_tweets['new_tweet']):
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         #deception_df.iat[i,j]=ss[k]
         ai_tweets.loc[index,ai_tweets.columns.isin([k])]=ss[k]
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()



# Now plot wor cloud for each tweet and see the sentiment score from sentment intesity analyzer
ai_tweets['length']=[len(item) for item in ai_tweets.new_tweet]
sorted_ai_tweets = ai_tweets.sort_values('length',ascending=False)[1:100:]


# Get word count
ai_tweets['word_count'] = ai_tweets['new_tweet'].apply(lambda x: (len(x.split())))
ai_tweets['word_count'].head()



plt.rcParams["figure.figsize"] = (10,5)
for index, tweet in enumerate(sorted_ai_tweets['new_tweet']):
    if len(tweet)<=3 :
        tweet="NoData"
    else:
        review=tweet
    fig, axes = plt.subplots(1, 2)
    wc=WordCloud(max_font_size=50, max_words=100, background_color="white",width = 512, height = 250,  mask=mask)
    wordcloud =wc.generate(tweet)
    #fig.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
#    plt.imshow(wordcloud, interpolation="bilinear")
    fig.suptitle("Tweet Number: "+str(index))
    axes[0].imshow(wc, interpolation="bilinear")
    if ai_tweets.loc[index,'pos']-ai_tweets.loc[index,'neg']>0:
        axes[1].imshow(mask_pos, cmap=plt.cm.gray, interpolation="bilinear")
    elif ai_tweets.loc[index,'pos']-ai_tweets.loc[index,'neg']<0:
        axes[1].imshow(mask_neg, cmap=plt.cm.gray, interpolation="bilinear")
    else:
        axes[1].imshow(mask_neu, cmap=plt.cm.gray, interpolation="bilinear")
    for ax in axes:
        ax.set_axis_off()
    plt.show()


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


#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in ai_tweets['new_tweet']:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


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



#  unigram boolean vectorizer, set minimum document frequency to 1
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=1)
vecs_bool=unigram_bool_vectorizer.fit_transform(ai_tweets['new_tweet'])


#  unigram term frequency vectorizer, set minimum document frequency to 1
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=1)
# fit vocabulary in documents and transform the documents into vectors
vecs_cnt = unigram_count_vectorizer.fit_transform(ai_tweets['new_tweet'])


#  unigram tfidf vectorizer, set minimum document frequency to 1
unigram_tf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=False, min_df=1)
vecs_tf= unigram_tf_vectorizer.fit_transform(ai_tweets['new_tweet'])

#  unigram tfidf vectorizer, set minimum document frequency to 1
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=1)
vecs_tfidf= unigram_tfidf_vectorizer.fit_transform(ai_tweets['new_tweet'])



# Using TfidfTransformer

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(vecs_cnt)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': unigram_count_vectorizer.get_feature_names(), 'weight': weights})

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

w=pd.DataFrame(vecs_bool.toarray(),columns=unigram_bool_vectorizer.get_feature_names())
x=pd.DataFrame(vecs_cnt.toarray(),columns=unigram_count_vectorizer.get_feature_names())
y=pd.DataFrame(vecs_tf.toarray(),columns=unigram_tf_vectorizer.get_feature_names())
z=pd.DataFrame(vecs_tfidf.toarray(),columns=unigram_tfidf_vectorizer.get_feature_names())

## Get the euclidean dist between Emma and Pride&Prejudice
#Using sklearn

dist = euclidean_distances(x)
print(np.round(dist,0))  #The dist between Emma and Pride is 3856

#Measure of distance that takes into account the
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(x)
print(np.round(cosdist,3))  #cos dist should be .02


#from sklearn.cluster import KMeans

num_clusters = 2

km = KMeans(n_clusters=num_clusters)

km.fit(cosdist)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


tweets = { 'tweet_num': list(ai_tweets.index),'cluster': clusters, 'tweet': list(ai_tweets.new_tweet)}

frame = pd.DataFrame(tweets, index = [clusters] , columns = ['tweet_num','cluster', 'tweet'])

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)

grouped = frame['tweet_num'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

grouped.mean() #average rank (1 to 100) per cluster

terms = x.columns



print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        if terms[ind] in totalvocab_stemmed :
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d tweets:" % i, end='')
    for tweet in frame.ix[i]['tweet'].values.tolist():
        print(' %s,' % tweet, end='')
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
cluster_colors = {0: '#1b9e77', 1: '#d95f02'}

#set up cluster names using a dict
cluster_names = {0: 'Cluster 1', 
                 1: 'Cluster 2'}

for x, y, name in zip(xs, ys, x.index):
    plt.scatter(x, y)
    plt.text(x, y, name)

plt.show()


#some ipython magic to show the matplotlib plots inline
#%matplotlib inline 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#create data frame that has the result of the MDS plus the cluster numbers and tweets
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, tweet=(tweets['tweet_num']))) 

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
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['tweet_num'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

##PLotting the relative distances in 3D
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(cosdist)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], df_cnt.index):
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
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, tweet=(tweets['tweet']))) 

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
    labels = [i for i in group.tweet]
    
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

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=x.index);

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

# several commonly used vectorizer setting
# The vectorizer can do "fit" and "transform"
# fit is a process to collect unique tokens into the vocabulary
# transform is a process to convert each document to vector based on the vocabulary
# These two processes can be done together using fit_transform(), or used individually: fit() or transform()

result_file="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week5\\Batch_3725691_batch_results.csv"

amt_results = pd.read_csv(result_file) 
# Preview the first 5 lines of the loaded data 
amt_results.head()

amt=pd.read_csv(result_file, dtype={'Input.text': str})

#def compute_majority_gold(results):
#    """
#    Get the TRUE items that the annotators agreed they are true
#    :param results: key to worker answers dictionary
#    :return: key to majority label dictionary
#    """
#    majority_gold = { key : np.argmax(np.bincount([1 if annotations[0] else 0 for worker, annotations
#                                                   in results[key].iteritems()]))
#                      for key in results.keys() }
#    
#    return majority_gold


def load_results(result_file):
    """
    Load the batch results from the CSV
    :param result_file: the batch results CSV file from MTurk
    :return: the workers and the answers
    """
    worker_answers = {}
    workers = set()
    table = pd.read_csv(result_file, dtype={'Input.text': str})
    
    for index, row in table.iterrows():
    
        #hit_id = row['HITId']
        worker_id = row['WorkerId']
    
        # Input fields
        p1 = row['Input.text']

    
        # Answer fields
        answer= row['Answer.sentiment.label'] if pd.isnull(row['Answer.sentiment.label'])==False else row['Answer.sentiment']
        #answer =  row['Answer.sentiment.label']
#        comment = row['Answer.comment']
    
        key = p1
    

        if key not in worker_answers.keys():
            worker_answers[key] = {}
    
        workers.add(worker_id)
    
        worker_answers[key][worker_id] = (answer)
    
    return workers, worker_answers


def cohens_kappa(results, workers):
    """
    Compute Cohen's Kappa on all workers that answered at least 5 HITs
    :param results:
    :return:
    """
    answers_per_worker = { worker_id : { key : results[key][worker_id] for key in results.keys()
                                         if worker_id in results[key] }
                           for worker_id in workers }
    answers_per_worker = { worker_id : answers for worker_id, answers in answers_per_worker.items()
                           if len(answers) >= 5 }
    curr_workers = answers_per_worker.keys()
    worker_pairs = [(worker1, worker2) for worker1 in curr_workers for worker2 in curr_workers if worker1 != worker2]
    
    label_index = { "Positive" : 1, "Neutral" : 0, "Negative" : -1, np.nan:0 }
    pairwise_kappa = { worker_id : { } for worker_id in answers_per_worker.keys() }
    
    # Compute pairwise Kappa
    for (worker1, worker2) in worker_pairs:
    
        mutual_hits = set(answers_per_worker[worker1].keys()).intersection(set(answers_per_worker[worker2].keys()))
        mutual_hits = set([hit for hit in mutual_hits if not pd.isnull(hit)])
    
        if len(mutual_hits) >= 5:
            
            worker1_labels = np.array([label_index[answers_per_worker[worker1][key]] for key in mutual_hits])
            worker2_labels = np.array([label_index[answers_per_worker[worker2][key]] for key in mutual_hits])
            curr_kappa = cohen_kappa_score(worker1_labels, worker2_labels)
            print('Worker1: ', worker1,' worker1_labels: ',worker1_labels,'\nWorker2: ',worker2, ' worker2_labels: ', worker2_labels, '\npairwise kappa: ',curr_kappa )
            if not math.isnan(curr_kappa):
                pairwise_kappa[worker1][worker2] = curr_kappa
                pairwise_kappa[worker2][worker1] = curr_kappa
    
    # Remove worker answers with low agreement to others
    workers_to_remove = set()
    
    for worker, kappas in pairwise_kappa.items():
        if np.mean(list(kappas.values())) < 0.1:
            print ('Removing %s' % worker)
            workers_to_remove.add(worker)
    
    kappa = np.mean([k for worker1 in pairwise_kappa.keys() for worker2, k in pairwise_kappa[worker1].items()
                     if not worker1 in workers_to_remove and not worker2 in workers_to_remove])
    
    # Return the average
    return kappa, workers_to_remove


"""
Analyze the results of AMT Worker's label on sentiment analysis
"""

args = docopt("""Analyze the results of AMT Worker's label on sentiment analysis
    <batch_result_file>    The MTurk batch results file
""")

batch_result_file = args['<batch_result_file>']

# Load the results
#workers, results, keys_by_bin = load_results(batch_result_file)
workers, results = load_results(result_file)

# Compute agreement between workers that answered at least 5 HITs
kappa, workers_to_remove = cohens_kappa(results, workers)
print ('Cohen\'s kappa=%.2f' % kappa)
num_hits_removed = 0

for key, key_worker_answers in results.items():
    for worker in list(key_worker_answers):
        if worker in workers_to_remove:
            num_hits_removed += len(key_worker_answers[worker]) if pd.isnull(key_worker_answers[worker])==False else 0
            key_worker_answers.pop(worker, None)

print ('Number of workers removed: %d, total HITs removed: %d' % (len(workers_to_remove), num_hits_removed))
print ()

## Compute accuracy for each bin
## keys_by_bin = { 1 : keys_by_bin[1] + keys_by_bin[2], 2 : keys_by_bin[3], 3 : keys_by_bin[4] }
#
#for bin in keys_by_bin.keys():
#
#    curr_results = { key : results[key] for key in keys_by_bin[bin] }
#    answers = compute_majority_gold(curr_results).values()
#    print ('Accuracy of bin %d: %.3f' % (bin, np.sum(answers) * 100.0 / len(curr_results)))
#
#print ()

