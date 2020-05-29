# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:47:23 2019

@author: rkrishnan
"""


from __future__ import division
#%matplotlib inline
import sys
# !{sys.executable} -m spacy download en
from pprint import pprint
import re
import collections
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer=SnowballStemmer( 'english')
from nltk.tokenize import WordPunctTokenizer
import os
from nltk.corpus import stopwords
from wordcloud import WordCloud
import gensim, spacy, logging, warnings
from gensim import models
import matplotlib.pyplot as plt
from gensim.utils import lemmatize, simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
path="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week9\\homework_8_data\\110"
## Print the files in this location to make sure I am where I want to be
print(os.listdir(path))
## Save the list
dir_list=os.listdir(path)
## get the dir list
print(dir_list)


def search_files(directory='.', extension=''):
    extension = extension.lower()
    file_path=[]
    file_name=[]
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                file_path.append(os.path.join(dirpath, name))
                file_name.append(os.path.join(name))
            elif not extension:
                file_path.append(os.path.join(dirpath, name))
                file_name.append(os.path.join(name))
    return [file_path,file_name]


def corpus_to_df(path,extension='.txt'):
    from nltk.corpus.reader import CategorizedPlaintextCorpusReader
    # Read corpus of document into dataframe
    # https://stackoverflow.com/questions/49088978/how-to-create-corpus-from-pandas-data-frame-to-operate-with-nltk
    my_corpus=CategorizedPlaintextCorpusReader(path,r'.*', cat_pattern=(r'*'+extension),encoding='cp1252') 
    my_corpus.fileids() # <- I expect values from column ID
    #my_corpus.words(fileids='110-m-r/110_sessions_x_tx.txt') # <- I expect values from column TITLE and BODY
    corpus_file_nm= pd.DataFrame(my_corpus.fileids(),columns=['file_nm'])
    bow,doc=[],[]
    for index in range(corpus_file_nm.count()[0]):
        doc.append(corpus_file_nm['file_nm'][index])
        my_list = my_corpus.words(fileids=corpus_file_nm['file_nm'][index])
        bow.append(my_list)
    
    data_df=pd.DataFrame()
    data_df['doc'] =doc
    data_df['bow'] =bow
    
    return data_df

file_path,file_name=search_files(path,'.txt')
file_df=pd.DataFrame(file_path,file_name,columns=['file_path'])
congress_df=corpus_to_df(path,'.txt')
## OK good - now I have a list of the filenames

for index, speech in congress_df.head(n=2).iterrows():
     print(index, speech)

# convert to lower case
congress_df['bow'] = congress_df['bow'].apply(lambda x: " ".join(x.lower() for x in x))
congress_df['bow'].head()

congress_df_pro=pd.DataFrame()
congress_df_pro=congress_df.head(5)
#remove stopwords

stop = stopwords.words('english')
for item in ["speaker","bill",",","$","),",",''","?''",":","ms","mr","mrs","the",">","<","doc","docno",".","(",")","))</","text","</","``","?","''",".","-",".''","--","'","alabama",
"alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia",
"hawaii","idaho","illinoisindiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts",
"michigan","minnesota","mississippi","missouri","montananebraska","nevada","new","hampshire","new","jersey",
"new","mexico","new","york","north","carolina","north","dakota","ohio","oklahoma","oregon","pennsylvania","rhode","island",
"southcarolina","southdakota","tennessee","texas","utah","vermont","virginia","washington","west","virginia",
"wisconsin","wyoming"]:
    stop.append(item)

for item in [name[1] for name in [(lambda x: x.split("_"))(x) for x in file_name]]:
    stop.append(item)
    
congress_df['bow'] = congress_df['bow'].apply(lambda x: " ".join(x for x in  x.split() if x not in stop))
congress_df['bow'].head()

for index, speech in enumerate(congress_df_pro['bow']):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate((speech))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(congress_df['doc'][index])
    plt.show()

congress_df_pro.loc[0,'bow']

word_pattern = re.compile("^\w+$")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def get_text_counter(text):
    tokens = WordPunctTokenizer().tokenize(text)
    tokens = list(map(lambda x: x.lower(), tokens))
    tokens = [lemmatize_stemming(token) for token in tokens if re.match(word_pattern, token) and token not in stop]
    return collections.Counter(tokens), len(tokens)

def make_df(counter, size):
    abs_freq = np.array([el[1] for el in counter])
    rel_freq = abs_freq / size
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq, rel_freq]).T, index=index, columns=["Absolute frequency", "Relative frequency"])
    df.index.name = "Most common words"
    return df

def tokenize(doc):
    words = []
    for word in doc.split(' '):
        words.append(word)
    return words

counter=[]
size=[]

for index,item in enumerate(congress_df['bow']):
    x_counter, x_size = get_text_counter(item)
    counter.append(x_counter)
    size.append(x_size)
    if index==0:
        all_counter =  x_counter
    else:
        all_counter = all_counter + x_counter
        
make_df(counter[0].most_common(10), size[0])

# check if the above loop worked fine by individually processing a file
y_counter, y_size = get_text_counter(congress_df_pro.loc[0,'bow'])
make_df(y_counter.most_common(10), y_size)

all_df = make_df(all_counter.most_common(), 1)
most_common_words = all_df.index.values


df_data = []
for word in most_common_words:
    x_c=[]
    for i in range(len(counter)):
        x_c.append(counter[i].get(word, 0) / size[i])
        #y_c = y_counter.get(word, 0) / y_size
        #d = abs(x_c - y_c)
    df_data.append(x_c)

dist_df = pd.DataFrame(data=df_data, index=most_common_words,
                       columns=file_name).T
#dist_df.index.name = "Most common words"
#dist_df.sort_values("Relative frequency difference", ascending=False, inplace=True)

dist_df.head(10)

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

congress_lst = congress_df['bow'].map(preprocess)
congress_lst[:10]



#https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

congress_dict = gensim.corpora.Dictionary(congress_lst)

count = 0
for k, v in congress_dict.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

congress_dict.filter_extremes(no_below=20, no_above=0.5, keep_n=100000)

bow_corpus = [congress_dict.doc2bow(doc) for doc in congress_lst]
bow_corpus[2]


bow_doc_2 = bow_corpus[2]
for i in range(len(bow_doc_2)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_2[i][0], 
                                               congress_dict[bow_doc_2[i][0]], 
bow_doc_2[i][1]))


tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model_viz = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=10, id2word=congress_dict, passes=2, random_state=100,
                                           chunksize=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=congress_dict, passes=2, workers=2, random_state=100,
                                           chunksize=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
       
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=congress_dict, passes=2, workers=4, random_state=100,
                                           chunksize=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

# Performance evaluation by classifying sample document using LDA Bag of Words model
for index, score in sorted(lda_model[bow_corpus[2]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
    
# Performance evaluation by classifying sample document using LDA TF-IDF model.

for index, score in sorted(lda_model_tfidf[bow_corpus[2]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

#Testing model on unseen document    
unseen_document = congress_df_pro.loc[3,'bow']
bow_vector = congress_dict.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    
################################################################
## sklearn
###################################################################3
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

lda_model_sk = LatentDirichletAllocation(n_components=10, max_iter=1000, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
congress_lda_sk = lda_model_sk.fit_transform(dist_df)

print("SIZE: ", congress_lda_sk.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in cogress data...")
print(congress_lda_sk[0])
print("Seventh Doc in congress Hike...")
print(congress_lda_sk[6])

## Print LDA using print function from above
print("LDA congress Model:")

## implement a print function 
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
 
print_topics(lda_model_sk, dist_df)

# Convert to list
data = congress_df.bow.values.tolist()

def format_topics_sentences(ldamodel=None, corpus=bow_corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus, texts=congress_lst)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)


doc_lens = [len(d) for d in df_dominant_topic.Text]

# Frequency Distribution of Word Counts in Documents
# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins = 1000, color='navy')
plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,1000,9))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()


import seaborn as sns
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):    
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 1000, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 1000), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i])
    ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,1000,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
plt.show()

# Word Clouds of Top N Keywords in Each Topic

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)


fig, axes = plt.subplots(4, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):

        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# Word Counts of Topic Keywords

from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in congress_lst for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(4, 3, figsize=(16,20), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    if (i<10):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()

# Sentence Chart Colored by Topic

# Sentence Coloring of N Sentences
from matplotlib.patches import Rectangle

def sentences_chart(lda_model=lda_model, corpus=bow_corpus, start = 0, end = 13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1] 
            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)       

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()

sentences_chart()  

# What are the most discussed topics in the documents?

# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=bow_corpus, end=-1)            

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)

from matplotlib.ticker import FuncFormatter

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), dpi=120, sharey=True)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
ax1.set_ylabel('Number of Documents')
ax1.set_ylim(0, 100)

# Topic Distribution by Topic Weights
ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

plt.show()

#t-SNE Clustering Chart
#Let’s visualize the clusters of documents in a 2D space using t-SNE (t-distributed stochastic neighbor embedding) algorithm

# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[bow_corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)


####################################################
##
## VISUALIZATION
##
####################################################
#https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
#https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_viz, bow_corpus, congress_dict)
vis