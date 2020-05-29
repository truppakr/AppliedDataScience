# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:38:16 2019

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

#import warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
from sklearn.cluster import KMeans
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from itertools import chain, groupby
import collections
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pandas as pd
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
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
############### Function to plot confusion matrix begins here ######################

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
    print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # ax.text(i, j, format(cm[i, j], fmt),
            ax.text(i, j, ("\n\n\n\n\n"+str(cm[i, j])) if j==0 else (str(cm[i, j])+"\n\n\n\n\n"),
                    ha="center", va="center",
                    color="red" if cm[i, j] > thresh else "red")
    fig.tight_layout()
    return ax
############### Function to plot confusion matrix ends here ######################



## I know that using"filename" with CountVectorizer requires a LIST
## of the file names. So I need to build one....

path="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week4\\"
print("calling os...")
## Print the files in this location to make sure I am where I want to be
print(os.listdir(path))
## Save the list
FileNameList=os.listdir(path)
## check the TYPE
print(type(FileNameList))
print(FileNameList)

filename="deception_data_converted_final.csv"
text = open(path+filename).readlines()
positive_lex=open(path+"opinion-lexicon-English\\positive-words.txt").readlines()
#import datetime
#todays_date = datetime.datetime.now().date()
#index = pd.date_range(todays_date-datetime.timedelta(10), periods=10, freq='D')
positive_lex_df=pd.DataFrame(columns=literal_eval(re.sub('\\\\n','',str(positive_lex[30:]))))
positive_lex_df.loc[0] =  list(([1]*len(positive_lex[30:])))

#posistive_lex=pd.DataFrame({'pos_lex':posistive_lex[30:]}).T
negative_lex=open(path+"opinion-lexicon-English\\negative-words.txt").readlines()
negative_lex_df=pd.DataFrame(columns=literal_eval(re.sub('\\\\n','',str(negative_lex[30:]))))
negative_lex_df.loc[0] =  list(([-1]*len(negative_lex[30:])))

deception_df=pd.DataFrame()
deception_df['lie']=""
deception_df['sentiment']=""
deception_df['review']=""
for index,line in enumerate(text):
    deception_df.loc[index-1,'lie']=str(text[index:index+1])[2:3]
    deception_df.loc[index-1,'sentiment']=str(text[index:index+1])[4:5]
    xx=re.sub('\\\\n','',str(text[index:index+1])[6:])
    deception_df.loc[index-1,'review']=re.sub('[\\\\"\]\']','',xx)
deception_df=deception_df[1:]

# convert to lower case
deception_df['review'] = deception_df['review'].apply(lambda x: "".join(x.lower() for x in x))
deception_df['review'].head()

#remove stopwords
stop = stopwords.words('english')
stop.append("the")
deception_df['review'] = deception_df['review'].apply(lambda x: " ".join(x for x in  x.split() if x not in stop))
deception_df['review'].head()

##correct spellings
#from textblob import TextBlob
#deception_df['review'] = deception_df['review'].apply(lambda x: str(TextBlob(x).correct()))
#deception_df['review'].head()


lie_df=deception_df.loc[:,deception_df.columns.isin(['lie','review'])]
sentiment_df=deception_df.iloc[:,1:3]


## read the mask / color image taken from
## http://jirkavinse.deviantart.com/art/quot-Real-Life-quot-Alice-282261010
## get data directory (using getcwd() is needed to support running example in generated IPython notebook)
#d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
#alice_coloring = np.array(Image.open(path.join(d, "alice_color.png")))
#wc = WordCloud(background_color="white", max_words=2000, mask=alice_coloring,
#               stopwords=stopwords, max_font_size=40, random_state=42)
## generate word cloud
#wc.generate(text)
#
## create coloring from image
#image_colors = ImageColorGenerator(alice_coloring)
#
## show
#fig, axes = plt.subplots(1, 3)
#axes[0].imshow(wc, interpolation="bilinear")
## recolor wordcloud and show
## we could also give color_func=image_colors directly in the constructor
#axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
#axes[2].imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
#for ax in axes:
#    ax.set_axis_off()
#plt.show()
mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/v/o/v/j/p/S/70-hi.png', stream=True).raw))
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


deception_df['compound']=0.00
deception_df['neg']=0.000
deception_df['neu']=0.000
deception_df['pos']=0.000
sid = SentimentIntensityAnalyzer()
for index,sentence in enumerate(deception_df['review']):
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         #deception_df.iat[i,j]=ss[k]
         deception_df.loc[index,deception_df.columns.isin([k])]=ss[k]
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()
plt.rcParams["figure.figsize"] = (5,5)
# Now plot wor cloud for each reviews and see the sentiment score from sentment intesity analyzer
for index, review in enumerate(deception_df['review']):
    if len(review)<=3 :
        review="NoData"
    else:
        review=review
    fig, axes = plt.subplots(1, 2)
    wc=WordCloud(max_font_size=50, max_words=100, background_color="white",width = 512, height = 512,  mask=mask)
    wordcloud =wc.generate(review)
    #fig.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
#    plt.imshow(wordcloud, interpolation="bilinear")
    fig.suptitle("Review Number: "+str(index))
    axes[0].imshow(wc, interpolation="bilinear")
    if deception_df.loc[index,'pos']-deception_df.loc[index,'neg']>0:
        axes[1].imshow(mask_pos, cmap=plt.cm.gray, interpolation="bilinear")
    elif deception_df.loc[index,'pos']-deception_df.loc[index,'neg']<0:
        axes[1].imshow(mask_neg, cmap=plt.cm.gray, interpolation="bilinear")
    else:
        axes[1].imshow(mask_neu, cmap=plt.cm.gray, interpolation="bilinear")
    for ax in axes:
        ax.set_axis_off()
    plt.show()


## CountVectorizers be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer

MyVect=CountVectorizer(input='content')
## NOw I can vectorize using my list of complete paths to my files
X_R=MyVect.fit_transform(deception_df['review'])
print(X_R)
## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNames=MyVect.get_feature_names()
## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF=pd.DataFrame(X_R.toarray(),columns=ColumnNames)
print(CorpusDF)
print(type(CorpusDF))
## We have what we expected - a data frame.
# Convert DataFrame to matrix
MyMatrix = CorpusDF.values
## Check it
print(type(MyMatrix))
print(MyMatrix)
## We need to "build" this in steps.
## First, I know that I need to be able to access
## the columns...
for name in ColumnNames:
    print(name)
## OK - that works...
## Now access the column by name
for name in ColumnNames:
    print(CorpusDF[name])
## OK - can we "add" columns??
print("The initial column names:\n", ColumnNames)
print(type(ColumnNames))  ## This is a list
MyStops=["also", "and", "are", "you", "of", "let", "not", "the", "for", "why", "there", "one", "which"]   


#import pandas as pd
# 
#df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12]],
#                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
#                  index=['Basket1', 'Basket2', 'Basket3'])
# 
#if 'Apple' in df.columns:
#    print("Yes")
#else:
#    print("No")
# 
# 
#if set(['Apple','Orange']).issubset(df.columns):
#    print("Yes")
#else:
#    print("No")
# 

 ## MAKE COPIES!
CleanDF=CorpusDF.copy()
print("START\n",CleanDF)
## Build a new columns list
ColNames=[]
for name in ColumnNames:
    #print("FFFFFFFF",name)
    if ((name in MyStops) or (len(name)<3) or str.isdigit(name) or str.isdigit(name.split("th")[0])  or str.isdigit(name.split("st")[0]) or  str.isdigit(name.split("nd")[0]) or str.isdigit(name.split("rd")[0]) ):
        #print("Dropping: ", name)
        CleanDF=CleanDF.drop([name], axis=1)
        #print(CleanDF)
    elif (lem.lemmatize(name,"v")!=name):
        if name in negative_lex_df.columns:
            #CleanDF[name]=CleanDF[name]*-1
            # Changing it back to positive number as the MNB does not take negative values
            CleanDF[name]=CleanDF[name]*1
            CleanDF=CleanDF.rename(columns={name: lem.lemmatize(name,"v")})
            ColNames.append(lem.lemmatize(name,"v"))

        else:

            CleanDF=CleanDF.rename(columns={name: lem.lemmatize(name,"v")})
            ColNames.append(lem.lemmatize(name,"v"))

    else:
        ## I MUST add these new ColNames
        if name in negative_lex_df.columns:
            # CleanDF[name]=CleanDF[name]*-1
            # Changing it back to positive number as the MNB does not take negative values
            CleanDF[name]=CleanDF[name]*1
            ColNames.append(name)
        else:
            ColNames.append(name)

#print("END\n",CleanDF)             
print("The ending column names:\n", ColNames)

# Convert DataFrame to matrix
MyMatrixClean = CleanDF.values
## Check it

print(type(MyMatrixClean))
print(MyMatrixClean)


# Vectorize Term Frequencies begins here
ColNames_DF =pd.DataFrame()
ColNames_DF['ColNames']=ColNames
ColNames_DF['cnt']=1
ColNames_DF=ColNames_DF.groupby(['ColNames']).sum()
ColNames_DF=(ColNames_DF[ColNames_DF['cnt']>1])
CleanDF=CleanDF.groupby(level=0, axis=1).sum()


CleanDF_TF=pd.DataFrame()
CleanDF_TF=CleanDF.copy()
CleanDF_TF['rowsum']=CleanDF_TF.astype(bool).sum(axis=1)
FP_WordCount=CleanDF_TF['rowsum']
sum_col=len(CleanDF_TF.columns)-1
CleanDF_TF=CleanDF_TF.iloc[:,:sum_col].div(CleanDF_TF["rowsum"], axis=0)
#CleanDF_TF=CleanDF_TF.drop(["rowsum"], axis=1)


# Convert DataFrame to matrix
CleanDF_TF=CleanDF_TF.fillna(0)
MyTFMatrixClean = CleanDF_TF.values
# Vectorize Term Frequencies ends here

# Vectorize TFIDF Frequencies begins here
IDF=pd.DataFrame((len(CleanDF_TF)/CleanDF_TF.astype(bool).sum(axis=0)).apply(lambda x: math.log(x))).T
CleanDF_TFIDF=CleanDF_TF*IDF.values
CleanDF_TFIDF['good']
# Convert DataFrame to matrix
CleanDF_TFIDF=CleanDF_TFIDF.fillna(0)
MyTFIDFMatrixClean = CleanDF_TFIDF.values
# Vectorize TFIDF Frequencies ends here

# Plot word counts by reviews
deception_df.loc[deception_df.lie == 't', 'lie_color'] = 'b' 
deception_df.loc[deception_df.lie == 'f', 'lie_color'] = 'r' 
deception_df.loc[deception_df.sentiment == 'p', 'sentiment_color'] = 'b' 
deception_df.loc[deception_df.sentiment == 'n', 'sentiment_color'] = 'r' 
deception_df['sentiment_as_size'] = deception_df.sentiment.str.replace(r'(^.*p.*$)', "250").str.replace(r'(^.*n.*$)', "125")
#deception_df.loc[deception_df.lie_color =='r','color'] = 1
#deception_df.loc[deception_df.lie_color =='b','color'] = 2
# plot the word counts
#df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
#ax = df.plot.bar(x='lab', y='val', rot=0)
fig = plt.figure(figsize=(40,10))
ax = FP_WordCount.plot.bar(x='index', y='rowsum',color=deception_df['lie_color'], rot=0)
#fig.add_subplot(111)
#bar = ax.bar(x=FP_WordCount.index.values, y=CleanDF.astype(bool).sum(axis=1), rot=0)
ax.set_title('Word Count on Reviews')
ax.set_xlabel('Review Number')
ax.set_ylabel('Count of words')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)

fig = plt.figure(figsize=(40,10))
ax = FP_WordCount.plot.bar(x='index', y='rowsum',color=deception_df['sentiment_color'], rot=0)
#fig.add_subplot(111)
#bar = ax.bar(x=FP_WordCount.index.values, y=CleanDF.astype(bool).sum(axis=1), rot=0)
ax.set_title('Word Count on Reviews')
ax.set_xlabel('Review Number')
ax.set_ylabel('Count of words')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)


## NOW - let's try k means again....
############################## k means ########################
# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object.fit(MyMatrixClean)
# Get cluster assignment labels
labels = kmeans_object.labels_
print("k-means with k = 3\n", labels)
# Format results as a DataFrame
Myresults = pd.DataFrame([deception_df.index,labels]).T
Myresults.columns=['Reviews','Cluster']
print("k means RESULTS\n", Myresults)


#Myresults['lie_color'] = deception_df['lie_color']

#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(Myresults.index,Myresults['Cluster'],c=deception_df['lie_color'] ,s= pd.to_numeric(deception_df['sentiment_as_size']))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Review Number')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)


# Kmeans  on Term Frequencies

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_tf_object = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_tf_object.fit(MyTFMatrixClean)
# Get cluster assignment labels
tf_labels = kmeans_tf_object.labels_
print("k-means with k = 2\n", tf_labels)
# Format results as a DataFrame
Myresults_tf = pd.DataFrame([CleanDF_TF.index,tf_labels]).T
Myresults_tf.columns=['Reviews','Cluster']
print("k means RESULTS\n", Myresults_tf)


#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(Myresults_tf['Reviews'],Myresults_tf['Cluster'],c=deception_df['lie_color'],s= pd.to_numeric(deception_df['sentiment_as_size']))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Review Numbers')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)


# Kmeans  on TFIDF Frequencies

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_tfidf_object = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_tfidf_object.fit(MyTFIDFMatrixClean)
# Get cluster assignment labels
tfidf_labels = kmeans_tfidf_object.labels_
print("k-means with k = 2\n", tfidf_labels)
# Format results as a DataFrame
Myresults_tfidf = pd.DataFrame([CleanDF_TFIDF.index,tfidf_labels]).T
Myresults_tfidf.columns=['Reviews','Cluster']
print("k means RESULTS\n", Myresults_tfidf)


#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(Myresults_tfidf['Reviews'],Myresults_tfidf['Cluster'],c=deception_df['lie_color'],s= pd.to_numeric(deception_df['sentiment_as_size']))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Review Numbers')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)


#  unigram and bigram term frequency vectorizer, set minimum document frequency to 1
gram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=1)
vecs = gram12_count_vectorizer.fit_transform(deception_df['review'])
print(vecs.shape)  
print(vecs[0].toarray())
print(len(gram12_count_vectorizer.vocabulary_))
print(list(gram12_count_vectorizer.vocabulary_.items())[:100])
print(gram12_count_vectorizer.vocabulary_.get('quality'))
print(gram12_count_vectorizer.vocabulary_.get('low'))
print(gram12_count_vectorizer.vocabulary_.get('quality low'))
#Get feature names
print(gram12_count_vectorizer.get_feature_names())
gram12_df=pd.DataFrame(vecs.toarray(),columns=gram12_count_vectorizer.get_feature_names())



#  unigram tfidf vectorizer, set minimum document frequency to 1
gram12_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', ngram_range=(1,2), use_idf=False, min_df=1)
vecs_tfidf= gram12_tfidf_vectorizer.fit_transform(deception_df['review'])
print(vecs_tfidf.shape)  
print(vecs_tfidf[0].toarray())
print(len(gram12_tfidf_vectorizer.vocabulary_))
print(list(gram12_tfidf_vectorizer.vocabulary_.items())[:100])
print(gram12_tfidf_vectorizer.vocabulary_.get('quality'))
print(gram12_tfidf_vectorizer.vocabulary_.get('low'))
print(gram12_tfidf_vectorizer.vocabulary_.get('quality low'))
#Get feature names
print(gram12_tfidf_vectorizer.get_feature_names())
gram12_tfidf=pd.DataFrame(vecs_tfidf.toarray(),columns=gram12_tfidf_vectorizer.get_feature_names())


##############################################################################
#################### Sentiment Prediction Begins here ##########################
##############################################################################

## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.

CleanDF['Label']=deception_df['sentiment']
TrainDF, TestDF = train_test_split(CleanDF, test_size=0.3)

##-----------------------------------------------------------------
##
## Now we have a training set and a testing set. 
print("The training set is:")
print(TrainDF)
print("The testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=pd.to_numeric(TestDF["Label"].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
print(TestLabels)
## remove labels
TestDF = TestDF.drop(["Label"], axis=1)
print(TestDF)

####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
## When you look up this model, you learn that it wants the 
## DF seperate from the labels
TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
print(TrainDF_nolabels)
TrainLabels= pd.to_numeric(TrainDF['Label'].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))

print(TrainLabels)
MyModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = MyModelNB.predict(TestDF)
print("The prediction from NB is:")
print(Prediction)
print("The actual labels are:")
print(TestLabels)
## confusion matrix
# from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("The confusion matrix is:")
print(cnf_matrix)

np.set_printoptions(precision=2)
plt.rcParams["figure.figsize"] = (5,5)
# Plot non-normalized confusion matrix
plot_confusion_matrix(TestLabels, Prediction, classes=np.array(['n','p']),
                      title='Confusion matrix')

plt.show()

metrics_df_sent = pd.DataFrame()

metrics_df_sent['model']=""
metrics_df_sent['accuracy_score']=0
metrics_df_sent['precision_score']=0
metrics_df_sent['recall_score']=0
metrics_df_sent['f1_score']=0

print(accuracy_score(TestLabels, Prediction)) 
print(precision_score(TestLabels, Prediction, average="weighted"))
print(recall_score(TestLabels, Prediction, average="weighted")) 
print(f1_score(TestLabels, Prediction, average="weighted"))
print(classification_report(TestLabels, Prediction))
metrics_df_sent.loc[0,'model']="MultinomialNB"
metrics_df_sent.loc[0,'accuracy_score']=accuracy_score(TestLabels, Prediction)
metrics_df_sent.loc[0,'precision_score']=precision_score(TestLabels, Prediction, average="weighted")
metrics_df_sent.loc[0,'recall_score']=recall_score(TestLabels, Prediction, average="weighted")
metrics_df_sent.loc[0,'f1_score']=f1_score(TestLabels, Prediction, average="weighted")


acc=pd.DataFrame()
acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction)
fig = plt.figure(figsize=(5,5))
ax = acc.plot.bar( y='accuracy',rot=0)
ax.set_title('MultinomialNB sentiment analysis accuracy')
ax.set_xlabel('MultinomialNB')
ax.set_ylabel('Accuracy in %')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)


### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB.predict_proba(TestDF),2))

#######################################################
### Bernoulli #########################################
#######################################################
### NOTE TO CLASS: This should use the Binary
## DF and is not correct - be sure to fix it :)
from sklearn.naive_bayes import BernoulliNB
BernModel = BernoulliNB()
BernModel.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction =BernModel.predict(TestDF)
print("Bernoulli prediction:\n", Prediction)
print("Actual:\n",TestLabels)

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("The confusion matrix is:")
print(cnf_matrix)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(TestLabels, Prediction, classes=np.array(['n','p']),
                      title='Confusion matrix')

plt.show()


print(accuracy_score(TestLabels, Prediction)) 
print(precision_score(TestLabels, Prediction, average="weighted"))
print(recall_score(TestLabels, Prediction, average="weighted")) 
print(f1_score(TestLabels, Prediction, average="weighted"))
print(classification_report(TestLabels, Prediction))
metrics_df_sent.loc[1,'model']="BernoulliNB"
metrics_df_sent.loc[1,'accuracy_score']=accuracy_score(TestLabels, Prediction)
metrics_df_sent.loc[1,'precision_score']=precision_score(TestLabels, Prediction, average="weighted")
metrics_df_sent.loc[1,'recall_score']=recall_score(TestLabels, Prediction, average="weighted")
metrics_df_sent.loc[1,'f1_score']=f1_score(TestLabels, Prediction, average="weighted")


acc=pd.DataFrame()
acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction)
fig = plt.figure(figsize=(5,5))
ax = acc.plot.bar( y='accuracy',rot=0)
ax.set_title('BernoulliNB sentiment analysis accuracy')
ax.set_xlabel('BernoulliNB')
ax.set_ylabel('Accuracy in %')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)


#############################################
###########  SVM ############################
#############################################
from sklearn.svm import LinearSVC
SVM_Model=LinearSVC(C=10)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction =SVM_Model.predict(TestDF)
print("SVM prediction:\n", Prediction)
print("Actual:\n",TestLabels)

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("The confusion matrix is:")
print(cnf_matrix)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(TestLabels, Prediction, classes=np.array(['n','p']),
                      title='Confusion matrix')

plt.show()


print(accuracy_score(TestLabels, Prediction)) 
print(precision_score(TestLabels, Prediction, average="weighted"))
print(recall_score(TestLabels, Prediction, average="weighted")) 
print(f1_score(TestLabels, Prediction, average="weighted"))
print(classification_report(TestLabels, Prediction))
metrics_df_sent.loc[2,'model']="SVM"
metrics_df_sent.loc[2,'accuracy_score']=accuracy_score(TestLabels, Prediction)
metrics_df_sent.loc[2,'precision_score']=precision_score(TestLabels, Prediction, average="weighted")
metrics_df_sent.loc[2,'recall_score']=recall_score(TestLabels, Prediction, average="weighted")
metrics_df_sent.loc[2,'f1_score']=f1_score(TestLabels, Prediction, average="weighted")

acc=pd.DataFrame()
acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction)
fig = plt.figure(figsize=(5,5))
ax = acc.plot.bar( y='accuracy',rot=0)
ax.set_title('SVM sentiment analysis accuracy')
ax.set_xlabel('SVM')
ax.set_ylabel('Accuracy in %')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)


## Make some fake data:
#n_series = 3
#n_observations = 4
#x = np.arange(n_observations)
#xlabels=['accuracy_score','precision_score','recall_score','f1_score']
#z = np.arange(len(xlabels))  # the label locations
## Plotting:
#
#fig, ax = plt.subplots(figsize=(20,5))
#
## Determine bar widths
#width_cluster = 0.7
#width_bar = width_cluster/n_series
#
#for n in range(n_series):
#    x_positions = x+(width_bar*n)-width_cluster/2
#    ax.bar(x_positions, metrics_df_sent.T.iloc[1:5,n], width_bar, align='edge')
#    
#
#ax.set_ylim(0,1)
#ax.set_title('Prediction Metrics Vs Models')
#ax.set_xlabel('Models')
#ax.set_xticks(z)
#ax.set_xticklabels(xlabels)
#ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')
#ax.legend()
#
#fig.tight_layout()
#
#plt.show()



plt.rcParams["figure.figsize"] = (20,10)
MultinomialNB = metrics_df_sent.T.iloc[1:5,0]
BernoulliNB =metrics_df_sent.T.iloc[1:5,1]
SVM = metrics_df_sent.T.iloc[1:5,2]
index = ['accuracy_score', 'precision_score', 'recall_score',
         'f1_score']
df = pd.DataFrame({'MultinomialNB': MultinomialNB,
                   'BernoulliNB': BernoulliNB,
                   'SVM': SVM}, index=index)
ax = df.plot.bar(rot=0)
#bar = ax.bar(x=df.index.values, y=MultinomialNB,height=1)

ax.set_ylim(0,1)
ax.set_title('Prediction Metrics Vs Models')
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')
plt.plot(ax)
plt.show()



##############################################################################
#################### Sentiment Prediction Ends here ##########################
##############################################################################

##############################################################################
#################### Lie Prediction Begins here ##############################
##############################################################################

CleanDF['Label']=deception_df['lie']
TrainDF, TestDF = train_test_split(CleanDF, test_size=0.3)

##-----------------------------------------------------------------
##
## Now we have a training set and a testing set. 
print("The training set is:")
print(TrainDF)
print("The testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=pd.to_numeric(TestDF["Label"].apply(lambda x: "".join('0' if x=='f' else '1' for x in x )))
print(TestLabels)
## remove labels
TestDF = TestDF.drop(["Label"], axis=1)
print(TestDF)

####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
## When you look up this model, you learn that it wants the 
## DF seperate from the labels
TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
print(TrainDF_nolabels)
TrainLabels= pd.to_numeric(TrainDF['Label'].apply(lambda x: "".join('0' if x=='f' else '1' for x in x )))

print(TrainLabels)
MyModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = MyModelNB.predict(TestDF)
print("The prediction from NB is:")
print(Prediction)
print("The actual labels are:")
print(TestLabels)
## confusion matrix
# from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("The confusion matrix is:")
print(cnf_matrix)

np.set_printoptions(precision=2)
plt.rcParams["figure.figsize"] = (5,5)
# Plot non-normalized confusion matrix
plot_confusion_matrix(TestLabels, Prediction, classes=np.array(['f','t']),
                      title='Confusion matrix')

plt.show()


metrics_df_lie = pd.DataFrame()

metrics_df_lie['model']=""
metrics_df_lie['accuracy_score']=0
metrics_df_lie['precision_score']=0
metrics_df_lie['recall_score']=0
metrics_df_lie['f1_score']=0

print(accuracy_score(TestLabels, Prediction)) 
print(precision_score(TestLabels, Prediction, average="weighted"))
print(recall_score(TestLabels, Prediction, average="weighted")) 
print(f1_score(TestLabels, Prediction, average="weighted"))
print(classification_report(TestLabels, Prediction))
metrics_df_lie.loc[0,'model']="MultinomialNB"
metrics_df_lie.loc[0,'accuracy_score']=accuracy_score(TestLabels, Prediction)
metrics_df_lie.loc[0,'precision_score']=precision_score(TestLabels, Prediction, average="weighted")
metrics_df_lie.loc[0,'recall_score']=recall_score(TestLabels, Prediction, average="weighted")
metrics_df_lie.loc[0,'f1_score']=f1_score(TestLabels, Prediction, average="weighted")

acc=pd.DataFrame()
acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction)
fig = plt.figure(figsize=(5,5))
ax = acc.plot.bar( y='accuracy',rot=0)
ax.set_title('MultinomialNB lie detection accuracy')
ax.set_xlabel('MultinomialNB')
ax.set_ylabel('Accuracy in %')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB.predict_proba(TestDF),2))

#######################################################
### Bernoulli #########################################
#######################################################
### NOTE TO CLASS: This should use the Binary
## DF and is not correct - be sure to fix it :)
from sklearn.naive_bayes import BernoulliNB
BernModel = BernoulliNB()
BernModel.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction =BernModel.predict(TestDF)
print("Bernoulli prediction:\n", Prediction)
print("Actual:\n",TestLabels)

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("The confusion matrix is:")
print(cnf_matrix)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(TestLabels, Prediction, classes=np.array(['f','t']),
                      title='Confusion matrix')

plt.show()


print(accuracy_score(TestLabels, Prediction)) 
print(precision_score(TestLabels, Prediction, average="weighted"))
print(recall_score(TestLabels, Prediction, average="weighted")) 
print(f1_score(TestLabels, Prediction, average="weighted"))
print(classification_report(TestLabels, Prediction))
metrics_df_lie.loc[1,'model']="BernoulliNB"
metrics_df_lie.loc[1,'accuracy_score']=accuracy_score(TestLabels, Prediction)
metrics_df_lie.loc[1,'precision_score']=precision_score(TestLabels, Prediction, average="weighted")
metrics_df_lie.loc[1,'recall_score']=recall_score(TestLabels, Prediction, average="weighted")
metrics_df_lie.loc[1,'f1_score']=f1_score(TestLabels, Prediction, average="weighted")

acc=pd.DataFrame()
acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction)
fig = plt.figure(figsize=(5,5))
ax = acc.plot.bar( y='accuracy',rot=0)
ax.set_title('BernoulliNB lie detection accuracy')
ax.set_xlabel('BernoulliNB')
ax.set_ylabel('Accuracy in %')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)


#############################################
###########  SVM ############################
#############################################
from sklearn.svm import LinearSVC
SVM_Model=LinearSVC(C=10)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction =SVM_Model.predict(TestDF)
print("SVM prediction:\n", Prediction)
print("Actual:\n",TestLabels)

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("The confusion matrix is:")
print(cnf_matrix)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(TestLabels, Prediction, classes=np.array(['f','t']),
                      title='Confusion matrix')

plt.show()


print(accuracy_score(TestLabels, Prediction)) 
print(precision_score(TestLabels, Prediction, average="weighted"))
print(recall_score(TestLabels, Prediction, average="weighted")) 
print(f1_score(TestLabels, Prediction, average="weighted"))
print(classification_report(TestLabels, Prediction))
metrics_df_lie.loc[2,'model']="SVM"
metrics_df_lie.loc[2,'accuracy_score']=accuracy_score(TestLabels, Prediction)
metrics_df_lie.loc[2,'precision_score']=precision_score(TestLabels, Prediction, average="weighted")
metrics_df_lie.loc[2,'recall_score']=recall_score(TestLabels, Prediction, average="weighted")
metrics_df_lie.loc[2,'f1_score']=f1_score(TestLabels, Prediction, average="weighted")

acc=pd.DataFrame()
acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction)
fig = plt.figure(figsize=(5,5))
ax = acc.plot.bar( y='accuracy',rot=0)
ax.set_title('SVM lie detection accuracy')
ax.set_xlabel('SVM')
ax.set_ylabel('Accuracy in %')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(ax)


plt.rcParams["figure.figsize"] = (20,10)
MultinomialNB = metrics_df_lie.T.iloc[1:5,0]
BernoulliNB =metrics_df_lie.T.iloc[1:5,1]
SVM = metrics_df_lie.T.iloc[1:5,2]
index = ['accuracy_score', 'precision_score', 'recall_score',
         'f1_score']
df = pd.DataFrame({'MultinomialNB': MultinomialNB,
                   'BernoulliNB': BernoulliNB,
                   'SVM': SVM}, index=index)
ax = df.plot.bar(rot=0)
#bar = ax.bar(x=df.index.values, y=MultinomialNB,height=1)

ax.set_ylim(0,1)
ax.set_title('Prediction Metrics Vs Models')
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')
plt.plot(ax)
plt.show()

##############################################################################
#################### Lie Prediction Ends here ################################
##############################################################################