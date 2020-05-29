# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:17:37 2019

@author: rkrishnan
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:38:16 2019

@author: rkrishnan
"""

#warnings.filterwarnings("ignore", category=DeprecationWarning)
#import nltk
#from sklearn.cluster import KMeans
#from nltk.probability import FreqDist
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus.reader import CategorizedPlaintextCorpusReader
#from itertools import chain, groupby
#import collections
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.svm import LinearSVC

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
ps =PorterStemmer()
from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
#import re
from PIL import Image
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from wordcloud import WordCloud #,  ImageColorGenerator
import csv




def read_tsv(path,filename):
    input_data=[]
    with open(path+filename) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      for row in reader:
        input_data.append(row)
    from pandas import DataFrame
    input_df= DataFrame.from_records(input_data[1:],columns=input_data[0])
    return input_df



def convert_lower_case_and_remove_stopwords(input_df,append_stop_lst,text_column_nm):
    # convert to lower case
    input_df[text_column_nm] = input_df[text_column_nm].apply(lambda x: "".join(x.lower() for x in x))
    input_df[text_column_nm].head(5)
    
    #remove stopwords
    stop = stopwords.words('english')
    for item in append_stop_lst:
        stop.append(item)
        
    input_df[text_column_nm] = input_df[text_column_nm].apply(lambda x: " ".join(x for x in  x.split() if x not in stop))
    input_df[text_column_nm].head(5)

    return input_df


def gen_text_concat(input_clean_df,text_column_nm):
    input_clean_df['space']=" "
    return (pd.DataFrame(input_clean_df.loc[:,[text_column_nm,'space']].values.sum(axis=1),columns=[text_column_nm])[text_column_nm].values.sum(axis=0))


def generate_word_clouds(input_clean_df,text_column_nm,input_text="",for_each_record="N",for_each_group="N",for_df="Y",record_wc_limit=5,title_col='index',group_by_col='index'):
    if ((for_each_record=="Y") | (for_each_record=="Yes")):
        for index, record in enumerate(input_clean_df[text_column_nm]):
            if index >= record_wc_limit:
                break
            else:
                if len(record)>1:
                    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate((record))
                    plt.figure()
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    if title_col=='index':
                        plt.title(input_clean_df.index[index])
                    else:
                         plt.title(input_clean_df[title_col][index])
                    plt.show()
    
    if ((for_each_group=="Y") | (for_each_group=="Yes")):
        unique_groups=pd.DataFrame(pd.unique(input_clean_df[group_by_col]),columns=[group_by_col])
        for index, record in enumerate(unique_groups[group_by_col]):
            if index >= record_wc_limit:
                break
            else:
                wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate((gen_text_concat(input_clean_df[input_clean_df[group_by_col]==record],text_column_nm)))
                plt.figure()
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(print(group_by_col," : ",unique_groups[group_by_col][index]))
                plt.show()
        
    if ((for_df=="Y") | (for_df=="Yes")): 
        input_clean_df['space']=" "
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(input_text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Input Data")
        plt.show()
        

## use sentiment intesity analyzer to get the sentiment score of alltweets collected

#def get_sentiment_intensity(input_clean_df,text_column_nm=text_column_nm):
#    input_clean_df['compound']=0.00
#    input_clean_df['neg']=0.000
#    input_clean_df['neu']=0.000
#    input_clean_df['pos']=0.000
#    sid = SentimentIntensityAnalyzer()
#    for index,sentence in enumerate(input_clean_df[text_column_nm]):
#         #print(sentence)
#         ss = sid.polarity_scores(sentence)
#         for k in sorted(ss):
#             #deception_df.iat[i,j]=ss[k]
#             input_clean_df.loc[index,input_clean_df.columns.isin([k])]=ss[k]
#             #print('{0}: {1}, '.format(k, ss[k]), end='')
#         #print()
#    return input_clean_df
#
#movie_review_clean_df_sent_int=get_sentiment_intensity(movie_review_clean_df)

def get_sentiment_intensity_by_group(input_clean_df,group_by_col,text_column_nm):
    unique_groups=pd.DataFrame(pd.unique(input_clean_df[group_by_col]),columns=[group_by_col])
    unique_groups['compound']=0.00
    unique_groups['neg']=0.000
    unique_groups['neu']=0.000
    unique_groups['pos']=0.000
    sid = SentimentIntensityAnalyzer()
    for index,sentence in enumerate(unique_groups[group_by_col]):
         #print(sentence)
         ss = sid.polarity_scores(gen_text_concat(input_clean_df[input_clean_df[group_by_col]==sentence],text_column_nm))
         for k in sorted(ss):
             #deception_df.iat[i,j]=ss[k]
             unique_groups.loc[index,unique_groups.columns.isin([k])]=ss[k]
             #print('{0}: {1}, '.format(k, ss[k]), end='')
         #print()
    return unique_groups


def genreate_word_cloud_with_sentiment_intensity(input_clean_df,sentiment_intensity,group_by_col,text_column_nm,record_wc_limit=5):
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
    #image_colors = ImageColorGenerator(mask)
    
    #fig, axes = plt.subplots(1, 3)
    #axes[0].imshow(wc, interpolation="bilinear")
    ## recolor wordcloud and show
    ## we could also give color_func=image_colors directly in the constructor
    #axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    #axes[2].imshow(mask, cmap=plt.cm.gray, interpolation="bilinear")
    #for ax in axes:
    #    ax.set_axis_off()
    #plt.show()
    
    #plt.rcParams["figure.figsize"] = (5,5)
    # Now plot wor cloud for each reviews and see the sentiment score from sentment intesity analyzer
       
    unique_groups=pd.DataFrame(pd.unique(input_clean_df[group_by_col]),columns=[group_by_col])
    
    for index, record in enumerate(unique_groups[group_by_col]):
        if index >= record_wc_limit:
            break
        else:
            fig, axes = plt.subplots(1, 2,figsize=(8,8))
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white",width = 512, height = 512,  mask=mask).generate((gen_text_concat(input_clean_df[input_clean_df[group_by_col]==record],text_column_nm)))
            fig.suptitle(print(group_by_col," : ",unique_groups[group_by_col][index]))
            axes[0].imshow(wordcloud, interpolation="bilinear")
            if sentiment_intensity.loc[index,'pos']-sentiment_intensity.loc[index,'neg']>0:
                axes[1].imshow(mask_pos, cmap=plt.cm.gray, interpolation="bilinear")
            elif sentiment_intensity.loc[index,'pos']-sentiment_intensity.loc[index,'neg']<0:
                axes[1].imshow(mask_neg, cmap=plt.cm.gray, interpolation="bilinear")
            else:
                axes[1].imshow(mask_neu, cmap=plt.cm.gray, interpolation="bilinear")
            for ax in axes:
                ax.set_axis_off()
            plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.GnBu):
    ############### Function to plot confusion matrix begins here ######################
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
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    thresh = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # ax.text(i, j, format(cm[i, j], fmt),
            # ax.text(i, j, ("\n\n\n\n\n"+str(cm[i, j])) if j==0 else (str(cm[i, j])+"\n\n\n\n\n"),
            ax.text(i, j, ("\n"+str(cm[i, j])) if j==0 else (str(cm[i, j])+"\n"),
                    ha="center", va="center",fontweight="bold",
                    color="darkred" if cm[i, j] >= thresh else "darkred")
    fig.tight_layout()
    return ax
############### Function to plot confusion matrix ends here ######################



def gen_multinomial_bernoulli_svm_sent_models(CleanDF,vectorization,classes,test_size):
    ############### Function to generate sentiment prediction models using Multinomial , Bernoulli and SVM for the same dataset and compare the results ######################
    """
    This function genereates sentiment prediction Multinomial, Bernoulli and SVM Model for the given dataset by taking test size as param.
    Also,compares the confusion metrics and classification report.
    """
    ##############################################################################
    #################### Sentiment Prediction Begins here ##########################
    ##############################################################################
    
    ## Create the testing set - grab a sample from the training set. 
    ## Be careful. Notice that right now, our train set is sorted by label.
    ## If your train set is large enough, you can take a random sample.
    
    TrainDF, TestDF = train_test_split(CleanDF, test_size=test_size)
    
    ##-----------------------------------------------------------------
    ##
    ## Now we have a training set and a testing set. 
    print("The training set is:")
    print(TrainDF)
    print("The testing set is:")
    print(TestDF)
    
    ## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
    ## Save labels
    #TestLabels=pd.to_numeric(TestDF["Label"].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
    TestLabels=pd.to_numeric(TestDF["Label"])
    print(TestLabels)
    ## remove labels
    TestDF = TestDF.drop(["Label"], axis=1)
    print(TestDF)
    
    ####################################################################
    ########################### MultinomialNB ##########################
    ####################################################################
    from sklearn.naive_bayes import MultinomialNB
    
    
    #https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
    #Create the modeler
    MyModelNB= MultinomialNB()

    ## DF seperate from the labels
    TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
    print(TrainDF_nolabels)
    #TrainLabels= pd.to_numeric(TrainDF['Label'].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
    TrainLabels=pd.to_numeric(TrainDF['Label'])
    print(TrainLabels)
    MyModelNB.fit(TrainDF_nolabels, TrainLabels)
    
    Prediction1 = MyModelNB.predict(TestDF)
    
    print("The prediction from MNB is:")
    print(Prediction1)
    print("The actual labels are:")
    print(TestLabels)
    
    ## confusion matrix
    cnf_matrix = confusion_matrix(TestLabels, Prediction1)
    print("The confusion matrix is:")
    print(cnf_matrix)
    
    np.set_printoptions(precision=2)
    plt.rcParams["figure.figsize"] = (5,5)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(TestLabels, Prediction1, classes=np.array(classes),
                          title='MNB Confusion matrix')
    
    plt.show()
    
    metrics_df_sent = pd.DataFrame()
    
    metrics_df_sent['model']=""
    metrics_df_sent['vectorization']=""
    metrics_df_sent['accuracy_score']=0
    metrics_df_sent['precision_score']=0
    metrics_df_sent['recall_score']=0
    metrics_df_sent['f1_score']=0
    
    print(accuracy_score(TestLabels, Prediction1)) 
    print(precision_score(TestLabels, Prediction1, average="weighted"))
    print(recall_score(TestLabels, Prediction1, average="weighted")) 
    print(f1_score(TestLabels, Prediction1, average="weighted"))
    print(classification_report(TestLabels, Prediction1))
    metrics_df_sent.loc[0,'model']="MultinomialNB"
    metrics_df_sent.loc[0,'vectorization']=vectorization
    metrics_df_sent.loc[0,'accuracy_score']=accuracy_score(TestLabels, Prediction1)
    metrics_df_sent.loc[0,'precision_score']=precision_score(TestLabels, Prediction1, average="weighted")
    metrics_df_sent.loc[0,'recall_score']=recall_score(TestLabels, Prediction1, average="weighted")
    metrics_df_sent.loc[0,'f1_score']=f1_score(TestLabels, Prediction1, average="weighted")
    
    
    acc=pd.DataFrame()
    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction1)
    #fig = plt.figure(figsize=(5,5))
    ax = acc.plot.bar( y='accuracy',rot=0)
    ax.set_title('MultinomialNB sentiment analysis accuracy')
    ax.set_xlabel('MultinomialNB')
    ax.set_ylabel('Accuracy in %')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ### prediction probabilities
    ## columns are the labels in alphabetical order
    ## The decinal in the matrix are the prob of being
    ## that label
    print(np.round(MyModelNB.predict_proba(TestDF),2))
    
    
    
    ####################################################################
    ########################### BernoulliNB ############################
    ####################################################################
    ### NOTE TO CLASS: This should use the Binary
    from sklearn.naive_bayes import BernoulliNB
    BernModel = BernoulliNB()
    BernModel.fit(TrainDF_nolabels, TrainLabels)
    #BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    Prediction2 =BernModel.predict(TestDF)
    print("Bernoulli prediction:\n", Prediction2)
    print("Actual:\n",TestLabels)
    
    cnf_matrix = confusion_matrix(TestLabels, Prediction2)
    print("The confusion matrix is:")
    print(cnf_matrix)
    
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(TestLabels, Prediction2, classes=np.array(classes),
                          title='Bernoulli Confusion matrix')
    
    plt.show()
    
    
    print(accuracy_score(TestLabels, Prediction2)) 
    print(precision_score(TestLabels, Prediction2, average="weighted"))
    print(recall_score(TestLabels, Prediction2, average="weighted")) 
    print(f1_score(TestLabels, Prediction2, average="weighted"))
    print(classification_report(TestLabels, Prediction2))
    metrics_df_sent.loc[1,'model']="BernoulliNB"
    metrics_df_sent.loc[1,'vectorization']=vectorization
    metrics_df_sent.loc[1,'accuracy_score']=accuracy_score(TestLabels, Prediction2)
    metrics_df_sent.loc[1,'precision_score']=precision_score(TestLabels, Prediction2, average="weighted")
    metrics_df_sent.loc[1,'recall_score']=recall_score(TestLabels, Prediction2, average="weighted")
    metrics_df_sent.loc[1,'f1_score']=f1_score(TestLabels, Prediction2, average="weighted")
    
    
    acc=pd.DataFrame()
    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction2)
    #fig = plt.figure(figsize=(5,5))
    ax = acc.plot.bar( y='accuracy',rot=0)
    ax.set_title('BernoulliNB sentiment analysis accuracy')
    ax.set_xlabel('BernoulliNB')
    ax.set_ylabel('Accuracy in %')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)


    ####################################################################
    ########################### SVM ####################################
    ####################################################################
    from sklearn.svm import LinearSVC
    SVM_Model=LinearSVC(C=10)
    SVM_Model.fit(TrainDF_nolabels, TrainLabels)
    Prediction3 =SVM_Model.predict(TestDF)
    print("SVM prediction:\n", Prediction3)
    print("Actual:\n",TestLabels)

    cnf_matrix = confusion_matrix(TestLabels, Prediction3)
    print("The confusion matrix is:")
    print(cnf_matrix)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(TestLabels, Prediction3, classes=np.array(classes),
                          title='SVM Confusion matrix')

    plt.show()


    print(accuracy_score(TestLabels, Prediction3)) 
    print(precision_score(TestLabels, Prediction3, average="weighted"))
    print(recall_score(TestLabels, Prediction3, average="weighted")) 
    print(f1_score(TestLabels, Prediction3, average="weighted"))
    print(classification_report(TestLabels, Prediction3))
    
    metrics_df_sent.loc[2,'model']="SVM"
    metrics_df_sent.loc[2,'vectorization']=vectorization
    metrics_df_sent.loc[2,'accuracy_score']=accuracy_score(TestLabels, Prediction3)
    metrics_df_sent.loc[2,'precision_score']=precision_score(TestLabels, Prediction3, average="weighted")
    metrics_df_sent.loc[2,'recall_score']=recall_score(TestLabels, Prediction3, average="weighted")
    metrics_df_sent.loc[2,'f1_score']=f1_score(TestLabels, Prediction3, average="weighted")
    
    
    acc=pd.DataFrame()
    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction3)
    #fig = plt.figure(figsize=(5,5))
    ax = acc.plot.bar( y='accuracy',rot=0)
    ax.set_title('SVM sentiment analysis accuracy')
    ax.set_xlabel('SVM')
    ax.set_ylabel('Accuracy in %')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
   
    plt.rcParams["figure.figsize"] = (10,5)
    MultinomialNB = metrics_df_sent.T.iloc[2:6,0]
    BernoulliNB =metrics_df_sent.T.iloc[2:6,1]
    SVM =metrics_df_sent.T.iloc[2:6,2]
    index = ['accuracy_score', 'precision_score', 'recall_score',
             'f1_score']
    df = pd.DataFrame({'MultinomialNB': MultinomialNB,
                       'BernoulliNB': BernoulliNB,
                       'SVM': SVM}, index=index)
    ax = df.plot.bar(rot=0)

    
    ax.set_ylim(0,1)
    ax.set_title('Prediction Metrics Vs Models')
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')
    ##############################################################################
    #################### Sentiment Prediction Ends here ##########################
    ##############################################################################
    
    return metrics_df_sent


def gen_svms_sent_models(CleanDF,vectorization,classes,test_size):
    ############### Function to generate sentiment prediction on different SVM models for the same dataset and compare the results ######################
    """
    This function genereates sentiment prediction on different SVM Models for the given dataset by taking test size as param.
    Also,compares the confusion metrics and classification report.
    """
    ##############################################################################
    #################### Sentiment Prediction Begins here ##########################
    ##############################################################################
    
    ## Create the testing set - grab a sample from the training set. 
    ## Be careful. Notice that right now, our train set is sorted by label.
    ## If your train set is large enough, you can take a random sample.

    CleanDF_label=pd.DataFrame(CleanDF['Label'])
    CleanDF_nolabels=CleanDF.drop(["Label"], axis=1)
    
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(CleanDF_nolabels)

    CleanDF['Label']=CleanDF_label['Label']

   
    TrainDF, TestDF = train_test_split(CleanDF, test_size=test_size)
    
    ##-----------------------------------------------------------------
    ##
    ## Now we have a training set and a testing set. 
    print("The training set is:")
    print(TrainDF)
    print("The testing set is:")
    print(TestDF)
    
    ## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
    ## Save labels
    #TestLabels=pd.to_numeric(TestDF["Label"].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
    TestLabels=pd.to_numeric(TestDF["Label"])
    print(TestLabels)
    ## remove labels
    TestDF = TestDF.drop(["Label"], axis=1)
    print(TestDF)

    #TrainLabels= pd.to_numeric(TrainDF['Label'].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
    TrainLabels=pd.to_numeric(TrainDF['Label'])
    print(TrainLabels)
    
    ## DF seperate from the labels
    TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
    print(TrainDF_nolabels)
    
   
    TrainDF_nolabels = scaling.transform(TrainDF_nolabels)
    TestDF = scaling.transform(TestDF)


    ####################################################################
    ########################### SVM Linear Kernel ######################
    ####################################################################
    
    #Create the modeler
    from sklearn.svm import LinearSVC
    SVM1_Model=LinearSVC(C=10)
  
    SVM1_Model.fit(TrainDF_nolabels, TrainLabels)
    Prediction1 = SVM1_Model.predict(TestDF)
    
    print("Linear SVM prediction:\n", Prediction1)
    print("Actual:\n",TestLabels)

    ## confusion matrix
    cnf_matrix = confusion_matrix(TestLabels, Prediction1)
    print("The confusion matrix is:")
    print(cnf_matrix)
    
    np.set_printoptions(precision=2)
    plt.rcParams["figure.figsize"] = (5,5)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(TestLabels, Prediction1, classes=np.array(classes),
                          title='Linear SVM Confusion matrix')
    
    plt.show()
    
    metrics_df_sent = pd.DataFrame()
    
    metrics_df_sent['model']=""
    metrics_df_sent['vectorization']=""
    metrics_df_sent['accuracy_score']=0
    metrics_df_sent['precision_score']=0
    metrics_df_sent['recall_score']=0
    metrics_df_sent['f1_score']=0
    
    print(accuracy_score(TestLabels, Prediction1)) 
    print(precision_score(TestLabels, Prediction1, average="weighted"))
    print(recall_score(TestLabels, Prediction1, average="weighted")) 
    print(f1_score(TestLabels, Prediction1, average="weighted"))
    print(classification_report(TestLabels, Prediction1))
    metrics_df_sent.loc[0,'model']="Linear SVM"
    metrics_df_sent.loc[0,'vectorization']=vectorization
    metrics_df_sent.loc[0,'accuracy_score']=accuracy_score(TestLabels, Prediction1)
    metrics_df_sent.loc[0,'precision_score']=precision_score(TestLabels, Prediction1, average="weighted")
    metrics_df_sent.loc[0,'recall_score']=recall_score(TestLabels, Prediction1, average="weighted")
    metrics_df_sent.loc[0,'f1_score']=f1_score(TestLabels, Prediction1, average="weighted")
    
    
    acc=pd.DataFrame()
    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction1)
    #fig = plt.figure(figsize=(5,5))
    ax = acc.plot.bar( y='accuracy',rot=0)
    ax.set_title('Linear SVM sentiment analysis accuracy')
    ax.set_xlabel('Linear SVM')
    ax.set_ylabel('Accuracy in %')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
#
#    ####################################################################
#    ########################### SVM RBF Kernel #########################
#    ####################################################################
#
#    from sklearn.svm import SVC
#    SVM2_Model=SVC(C=10,kernel='rbf',gamma='auto')
#    SVM2_Model.fit(TrainDF_nolabels, TrainLabels)
#
#    Prediction2 =SVM2_Model.predict(TestDF)
#    print("SVM RBF Kernel prediction:\n", Prediction2)
#    print("Actual:\n",TestLabels)
#    
#    cnf_matrix = confusion_matrix(TestLabels, Prediction2)
#    print("The confusion matrix is:")
#    print(cnf_matrix)
#    
#    np.set_printoptions(precision=2)
#    
#    # Plot non-normalized confusion matrix
#    plot_confusion_matrix(TestLabels, Prediction2, classes=np.array(classes),
#                          title='SVM RBF Kernel Confusion matrix')
#    
#    plt.show()
#    
#    
#    print(accuracy_score(TestLabels, Prediction2)) 
#    print(precision_score(TestLabels, Prediction2, average="weighted"))
#    print(recall_score(TestLabels, Prediction2, average="weighted")) 
#    print(f1_score(TestLabels, Prediction2, average="weighted"))
#    print(classification_report(TestLabels, Prediction2))
#    metrics_df_sent.loc[1,'model']="SVM RBF Kernel"
#    metrics_df_sent.loc[1,'vectorization']=vectorization
#    metrics_df_sent.loc[1,'accuracy_score']=accuracy_score(TestLabels, Prediction2)
#    metrics_df_sent.loc[1,'precision_score']=precision_score(TestLabels, Prediction2, average="weighted")
#    metrics_df_sent.loc[1,'recall_score']=recall_score(TestLabels, Prediction2, average="weighted")
#    metrics_df_sent.loc[1,'f1_score']=f1_score(TestLabels, Prediction2, average="weighted")
#    
#    
#    acc=pd.DataFrame()
#    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction2)
#    #fig = plt.figure(figsize=(5,5))
#    ax = acc.plot.bar( y='accuracy',rot=0)
#    ax.set_title('SVM RBF Kernel sentiment analysis accuracy')
#    ax.set_xlabel('SVM RBF Kernel')
#    ax.set_ylabel('Accuracy in %')
#    for tick in ax.get_xticklabels():
#        tick.set_rotation(90)


    ####################################################################
    ########################### SVM Poly Kernel#########################
    ####################################################################
    from sklearn.svm import SVC
    SVM3_Model=SVC(C=10,kernel='poly',gamma='auto')
    SVM3_Model.fit(TrainDF_nolabels, TrainLabels)
    Prediction3 =SVM3_Model.predict(TestDF)
    print("SVM Poly Kernel prediction:\n", Prediction3)
    print("Actual:\n",TestLabels)

    cnf_matrix = confusion_matrix(TestLabels, Prediction3)
    print("The confusion matrix is:")
    print(cnf_matrix)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(TestLabels, Prediction3, classes=np.array(classes),
                          title='SVM Poly Kernel Confusion matrix')

    plt.show()


    print(accuracy_score(TestLabels, Prediction3)) 
    print(precision_score(TestLabels, Prediction3, average="weighted"))
    print(recall_score(TestLabels, Prediction3, average="weighted")) 
    print(f1_score(TestLabels, Prediction3, average="weighted"))
    print(classification_report(TestLabels, Prediction3))
    
    metrics_df_sent.loc[1,'model']="SVM Poly Kernel"
    metrics_df_sent.loc[1,'vectorization']=vectorization
    metrics_df_sent.loc[1,'accuracy_score']=accuracy_score(TestLabels, Prediction3)
    metrics_df_sent.loc[1,'precision_score']=precision_score(TestLabels, Prediction3, average="weighted")
    metrics_df_sent.loc[1,'recall_score']=recall_score(TestLabels, Prediction3, average="weighted")
    metrics_df_sent.loc[1,'f1_score']=f1_score(TestLabels, Prediction3, average="weighted")
    
    
    acc=pd.DataFrame()
    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction3)
    #fig = plt.figure(figsize=(5,5))
    ax = acc.plot.bar( y='accuracy',rot=0)
    ax.set_title('SVM Poly Kernel sentiment analysis accuracy')
    ax.set_xlabel('SVM Poly Kernel')
    ax.set_ylabel('Accuracy in %')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
    ####################################################################
    ########################### SVM Sigmoid Kernel#########################
    ####################################################################
    from sklearn.svm import SVC
    SVM4_Model=SVC(C=10,kernel='sigmoid',gamma='auto')
    SVM4_Model.fit(TrainDF_nolabels, TrainLabels)
    Prediction4 =SVM4_Model.predict(TestDF)
    print("SVM Sigmoid Kernel prediction:\n", Prediction4)
    print("Actual:\n",TestLabels)

    cnf_matrix = confusion_matrix(TestLabels, Prediction4)
    print("The confusion matrix is:")
    print(cnf_matrix)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(TestLabels, Prediction4, classes=np.array(classes),
                          title='SVM Sigmoid Kernel Confusion matrix')

    plt.show()


    print(accuracy_score(TestLabels, Prediction4)) 
    print(precision_score(TestLabels, Prediction4, average="weighted"))
    print(recall_score(TestLabels, Prediction4, average="weighted")) 
    print(f1_score(TestLabels, Prediction4, average="weighted"))
    print(classification_report(TestLabels, Prediction4))
    
    metrics_df_sent.loc[2,'model']="SVM Sigmoid Kernel"
    metrics_df_sent.loc[2,'vectorization']=vectorization
    metrics_df_sent.loc[2,'accuracy_score']=accuracy_score(TestLabels, Prediction4)
    metrics_df_sent.loc[2,'precision_score']=precision_score(TestLabels, Prediction4, average="weighted")
    metrics_df_sent.loc[2,'recall_score']=recall_score(TestLabels, Prediction4, average="weighted")
    metrics_df_sent.loc[2,'f1_score']=f1_score(TestLabels, Prediction4, average="weighted")
    
    
    acc=pd.DataFrame()
    acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction4)
    #fig = plt.figure(figsize=(5,5))
    ax = acc.plot.bar( y='accuracy',rot=0)
    ax.set_title('SVM Sigmoid Kernel sentiment analysis accuracy')
    ax.set_xlabel('SVM Sigmoid Kernel')
    ax.set_ylabel('Accuracy in %')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
   
    plt.rcParams["figure.figsize"] = (10,5)
    SVM1 =metrics_df_sent.T.iloc[2:6,0]
    # SVM2 =metrics_df_sent.T.iloc[2:6,1]
    SVM3 =metrics_df_sent.T.iloc[2:6,1]
    SVM4 =metrics_df_sent.T.iloc[2:6,2]
    index = ['accuracy_score', 'precision_score', 'recall_score',
             'f1_score']
    df = pd.DataFrame({'SVM_Linear': SVM1,
                       #'SVM_RBF': SVM2,
                       'SVM_Poly': SVM3,
                       'SVM_Sigmoid': SVM4}, index=index)
    ax = df.plot.bar(rot=0)

    
    ax.set_ylim(0,1)
    ax.set_title('Prediction Metrics Vs Models')
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')
    ##############################################################################
    #################### Sentiment Prediction Ends here ##########################
    ##############################################################################
    
    return metrics_df_sent

     
def get_linear_svm(CleanDF,vectorization,classes,test_size):
     
    ############### Function to generate sentiment prediction models using Multinomial , Bernoulli and SVM for the same dataset and compare the results ######################
    """
    This function genereates sentiment prediction Multinomial, Bernoulli and SVM Model for the given dataset by taking test size as param.
    Also,compares the confusion metrics and classification report.
    """
    ##############################################################################
    #################### Sentiment Prediction Begins here ##########################
    ##############################################################################
    
    ## Create the testing set - grab a sample from the training set. 
    ## Be careful. Notice that right now, our train set is sorted by label.
    ## If your train set is large enough, you can take a random sample.
    
    TrainDF, TestDF = train_test_split(CleanDF, test_size=test_size)
    
    ##-----------------------------------------------------------------
    ##
    ## Now we have a training set and a testing set. 
    print("The training set is:")
    print(TrainDF)
    print("The testing set is:")
    print(TestDF)
    
    ## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
    ## Save labels
    #TestLabels=pd.to_numeric(TestDF["Label"].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
    TestLabels=pd.to_numeric(TestDF["Label"])
    print(TestLabels)
    ## remove labels
    TestDF = TestDF.drop(["Label"], axis=1)
    print(TestDF)

    ## DF seperate from the labels
    TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
    print(TrainDF_nolabels)
    #TrainLabels= pd.to_numeric(TrainDF['Label'].apply(lambda x: "".join('0' if x=='n' else '1' for x in x )))
    TrainLabels=pd.to_numeric(TrainDF['Label'])
    print(TrainLabels)
    
    metrics_df_sent = pd.DataFrame()
    metrics_df_sent['model']=""
    metrics_df_sent['vectorization']=""
    metrics_df_sent['accuracy_score']=0
    metrics_df_sent['precision_score']=0
    metrics_df_sent['recall_score']=0
    metrics_df_sent['f1_score']=0

    ####################################################################
    ########################### SVM ####################################
    ####################################################################
    from sklearn.svm import LinearSVC
         
    cs = [0.1, 1, 10, 100, 1000]
    for index,c in enumerate(cs):

        SVM_Model=LinearSVC(C=c)
        SVM_Model.fit(TrainDF_nolabels, TrainLabels)
        Prediction3 =SVM_Model.predict(TestDF)
        print("SVM prediction:\n", Prediction3)
        print("Actual:\n",TestLabels)
    
        cnf_matrix = confusion_matrix(TestLabels, Prediction3)
        print("The confusion matrix is:")
        print(cnf_matrix)
    
        np.set_printoptions(precision=2)
    
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(TestLabels, Prediction3, classes=np.array(classes),
                              title=print('SVM Confusion matrix ( C=',str(c),' )'))
        plt.show()
    
        #plotSVC(‘C=’ + str(c))
        
        print(accuracy_score(TestLabels, Prediction3)) 
        print(precision_score(TestLabels, Prediction3, average="weighted"))
        print(recall_score(TestLabels, Prediction3, average="weighted")) 
        print(f1_score(TestLabels, Prediction3, average="weighted"))
        print(classification_report(TestLabels, Prediction3))
        
        metrics_df_sent.loc[index,'model']=print('SVM ( C=',str(c),' )')
        metrics_df_sent.loc[index,'vectorization']=vectorization
        metrics_df_sent.loc[index,'accuracy_score']=accuracy_score(TestLabels, Prediction3)
        metrics_df_sent.loc[index,'precision_score']=precision_score(TestLabels, Prediction3, average="weighted")
        metrics_df_sent.loc[index,'recall_score']=recall_score(TestLabels, Prediction3, average="weighted")
        metrics_df_sent.loc[index,'f1_score']=f1_score(TestLabels, Prediction3, average="weighted")
        
        
        acc=pd.DataFrame()
        acc.loc[0,'accuracy']=accuracy_score(TestLabels, Prediction3)
        #fig = plt.figure(figsize=(5,5))
        ax = acc.plot.bar( y='accuracy',rot=0)
        ax.set_title('SVM sentiment analysis accuracy')
        ax.set_xlabel('SVM')
        ax.set_ylabel('Accuracy in %')
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        
       
    plt.rcParams["figure.figsize"] = (10,5)
    SVM1 = metrics_df_sent.T.iloc[2:6,0]
    SVM2 =metrics_df_sent.T.iloc[2:6,1]
    SVM3 =metrics_df_sent.T.iloc[2:6,2]
    SVM4 =metrics_df_sent.T.iloc[2:6,3]
    SVM5 =metrics_df_sent.T.iloc[2:6,4]
    index = ['accuracy_score', 'precision_score', 'recall_score',
             'f1_score']
    df = pd.DataFrame({'SVM C=0.1': SVM1,
                       'SVM C=1': SVM2,
                       'SVM C=10': SVM3,
                       'SVM C=100': SVM4,
                       'SVM C=1000': SVM5}, index=index)
    ax = df.plot.bar(rot=0)
    ax.set_ylim(0,1)
    ax.set_title('Prediction Metrics Vs Models')
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')
    ##############################################################################
    #################### Sentiment Prediction Ends here ##########################
    ##############################################################################

    return metrics_df_sent


path="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week7\\sentiment-analysis-on-movie-reviews\\"
filename="train.tsv"
movie_review_df=  read_tsv(path=path,filename=filename)
append_stop_lst=[]
text_column_nm="Phrase"
movie_review_clean_df = convert_lower_case_and_remove_stopwords(movie_review_df,append_stop_lst,text_column_nm)
movie_review_text=gen_text_concat(movie_review_clean_df.head(20),text_column_nm)
generate_word_clouds(movie_review_clean_df,text_column_nm,input_text=movie_review_text,for_each_record="N",for_each_group="Y",for_df="Y",record_wc_limit=2,title_col='index',group_by_col='SentenceId')
movie_review_group_sent_int=get_sentiment_intensity_by_group(movie_review_clean_df,text_column_nm=text_column_nm,group_by_col="SentenceId")
classes=list(pd.unique(movie_review_clean_df['Sentiment']))
#genreate_word_cloud_with_sentiment_intensity(movie_review_clean_df,movie_review_group_sent_int,text_column_nm=text_column_nm,group_by_col="SentenceId")


#  unigram boolean vectorizer, set minimum document frequency to 50
# fit vocabulary in documents and transform the documents into vectors
metrics_df_sent=pd.DataFrame()
metrics_df_sent['model']=""
metrics_df_sent['vectorization']=""
metrics_df_sent['accuracy_score']=0
metrics_df_sent['precision_score']=0
metrics_df_sent['recall_score']=0
metrics_df_sent['f1_score']=0


unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=50)
vecs_bool=unigram_bool_vectorizer.fit_transform(movie_review_clean_df[text_column_nm])
print(vecs_bool.shape)
print(len(unigram_bool_vectorizer.vocabulary_))
# print(list(unigram_bool_vectorizer.vocabulary_.items())[:100])
# print(unigram_bool_vectorizer.vocabulary_.get('bad'))
# print(unigram_bool_vectorizer.get_feature_names())
df_vecs_bool=pd.DataFrame(vecs_bool.toarray(),columns=unigram_bool_vectorizer.get_feature_names())
df_vecs_bool['Label']=movie_review_clean_df['Sentiment']
metrics_df_sent=metrics_df_sent[metrics_df_sent.vectorization != 'Unigram_Boolean']
metrics_df_sent=gen_multinomial_bernoulli_svm_sent_models(df_vecs_bool,'Unigram_Boolean',classes,test_size=0.3)


#  unigram term frequency vectorizer, set minimum document frequency to 50
# fit vocabulary in documents and transform the documents into vectors
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=50)
vecs_cnt = unigram_count_vectorizer.fit_transform(movie_review_clean_df[text_column_nm])
print(vecs_bool.shape)
print(len(unigram_bool_vectorizer.vocabulary_))
df_vecs_cnt=pd.DataFrame(vecs_cnt.toarray(),columns=unigram_count_vectorizer.get_feature_names())
df_vecs_cnt['Label']=movie_review_clean_df['Sentiment']
metrics_df_sent=metrics_df_sent[metrics_df_sent.vectorization != 'Unigram_Frequency']
metrics_df_sent=pd.concat([metrics_df_sent,gen_multinomial_bernoulli_svm_sent_models(df_vecs_cnt,'Unigram_Frequency',classes,test_size=0.4)])


#  grame 1,2 boolean vectorizer, set minimum document frequency to 50
# fit vocabulary in documents and transform the documents into vectors
gram12_bool_vectorizer = CountVectorizer(encoding='latin-1',ngram_range=(1,2), binary=True, min_df=50)
gram12_vecs_bool=gram12_bool_vectorizer.fit_transform(movie_review_clean_df[text_column_nm])
print(gram12_vecs_bool.shape)
print(len(gram12_bool_vectorizer.vocabulary_))
df_gram12_vecs_bool=pd.DataFrame(gram12_vecs_bool.toarray(),columns=gram12_bool_vectorizer.get_feature_names())
df_gram12_vecs_bool['Label']=movie_review_clean_df['Sentiment']
metrics_df_sent=metrics_df_sent[metrics_df_sent.vectorization != 'Gram12_Boolean']
metrics_df_sent=pd.concat([metrics_df_sent,gen_multinomial_bernoulli_svm_sent_models(df_gram12_vecs_bool,'Gram12_Boolean',classes,test_size=0.4)])


#  grame 1,2 count vectorizer, set minimum document frequency to 50
# fit vocabulary in documents and transform the documents into vectors



gram12_count_vectorizer = CountVectorizer(encoding='latin-1',ngram_range=(1,2), binary=False, min_df=50)
gram12_vecs_cnt = gram12_count_vectorizer.fit_transform(movie_review_clean_df[text_column_nm])
print(gram12_vecs_cnt.shape)  
print(len(gram12_count_vectorizer.vocabulary_))
df_gram12_vecs_cnt=pd.DataFrame(gram12_vecs_cnt.toarray(),columns=gram12_count_vectorizer.get_feature_names())

df_gram12_vecs_cnt['Label']=movie_review_clean_df['Sentiment']
metrics_df_sent=metrics_df_sent[metrics_df_sent.vectorization != 'Gram12_Frequency']
metrics_df_sent=pd.concat([metrics_df_sent,gen_multinomial_bernoulli_svm_sent_models(df_gram12_vecs_cnt,'Gram12_Frequency',classes,test_size=0.4)])


metrics_df_sent_plt=metrics_df_sent.reset_index(drop=True)

plt.rcParams["figure.figsize"] = (40,10)

nrow=2
ncol=int(len(np.unique(metrics_df_sent_plt.vectorization))/nrow)
num=0
r=0
c=0
# Find the right spot on the plot
fig = plt.figure(figsize=(28,14))
#    plt.subplot(2,4, num)
fig, axes = plt.subplots(nrow, ncol)
for index,vector in enumerate(np.unique(metrics_df_sent_plt.vectorization)):
    num+=1
    if(r==0 & c==0):
        r=1
        c=1
    elif ((0==(c)%2) & (c!=1)):
        c=1
        r+=1
    else:
        c=c+1
    print(r,c)
    MultinomialNB = metrics_df_sent_plt[metrics_df_sent_plt.vectorization==vector].T.iloc[2:6,0]
    BernoulliNB =metrics_df_sent_plt[metrics_df_sent_plt.vectorization==vector].T.iloc[2:6,1]
    SVM =metrics_df_sent_plt[metrics_df_sent_plt.vectorization==vector].T.iloc[2:6,2]
    index = ['accuracy_score', 'precision_score', 'recall_score',
             'f1_score']
    df = pd.DataFrame({'MultinomialNB': MultinomialNB,
                       'BernoulliNB': BernoulliNB,
                       'SVM': SVM}, index=index)
    ax=df.plot.bar(ax=axes[r-1,c-1],rot=0)
    #bar = ax.bar(x=df.index.values, y=MultinomialNB,height=1)
    ax.set_ylim(0,1)
    # ax.set_title('Prediction Metrics Vs Models')
    ax.set_xlabel(vector)
    ax.set_ylabel('Accuracy/Precision/Recall/f1-score in %')


plt.rcParams["figure.figsize"] = (10,5)
MultinomialNB =list(metrics_df_sent_plt[metrics_df_sent_plt.model=='MultinomialNB'].loc[0:,'accuracy_score'])
BernoulliNB =list(metrics_df_sent_plt[metrics_df_sent_plt.model=='BernoulliNB'].loc[0:,'accuracy_score'])
SVM =list(metrics_df_sent_plt[metrics_df_sent_plt.model=='SVM'].loc[0:,'accuracy_score'])
index =list(metrics_df_sent_plt[metrics_df_sent_plt.model=='MultinomialNB'].loc[0:,'vectorization'])
df1 = pd.DataFrame({'MultinomialNB': MultinomialNB,
                   'BernoulliNB': BernoulliNB,
                   'SVM': SVM}, index=index)
ax=df1.plot.bar(rot=45)
ax.set_ylim(0.4,1)
ax.set_title('Accuracy Metric Vs Different Models')
ax.set_xlabel('Sentiment Prediction Models')
ax.set_ylabel('Accuracy in %')


#  unigram boolean vectorizer, set minimum document frequency to 50
# fit vocabulary in documents and transform the documents into vectors
metrics_svms_df_sent=pd.DataFrame()
metrics_svms_df_sent['model']=""
metrics_svms_df_sent['vectorization']=""
metrics_svms_df_sent['accuracy_score']=0
metrics_svms_df_sent['precision_score']=0
metrics_svms_df_sent['recall_score']=0
metrics_svms_df_sent['f1_score']=0


#  grame 1,2 count vectorizer, set minimum document frequency to 50
# fit vocabulary in documents and transform the documents into vectors
metrics_svms_df_sent=metrics_svms_df_sent[metrics_svms_df_sent.vectorization != 'Gram12_Frequency']
metrics_svms_df_sent=get_linear_svm(df_gram12_vecs_cnt,'Gram12_Frequency',classes,test_size=0.4)

