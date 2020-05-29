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
###
### Gates
###
### YOU will need to CREATE things to use this code
### 1) Create a folder (corpus) of text files. 
###    Make text files very short and very topic 
###    specific. I have 3 .txt files on dogs, 3 on hiking
###    and one on both dogs and hiking. 
### 2) Use the same exact data to create csv file
###    where each row is a .txt file. Have labels.
####################################################
import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
ps =PorterStemmer()
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from itertools import chain, groupby
import collections
import math
from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#MyCorpusReader = nltk.corpus.PlaintextCorpusReader('CorpusHikeDog_Small', '.*\.txt')
#print(MyCorpusReader)
#print(MyCorpusReader.raw())
###BUT.. the above is not what I want. So I will write my own methods...

## I know that using"filename" with CountVectorizer requires a LIST
## of the file names. So I need to build one....

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
                
path="C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST 736\\Week9\\homework_8_data\\110"
print("calling os...")
## Print the files in this location to make sure I am where I want to be
print(os.listdir(path))

file_path,file_name=search_files(path,'.txt')

pd.DataFrame(file_path,file_name,)
## OK good - now I have a list of the filenames

## While one would think that this would just work directly - it will not
## WHen using CountVecotrizer with "filename", you need the COMPLETE PATHS
## So now that I have the list of file names in my corpus, I will build
## a list of complete paths to each....

## I need an empty list to start with:
## Notice that I defined path above.
ListOfCompleteFilePaths=[]
ListOfJustFileNames=[]
for name in os.listdir(path):
    print(path+ "\\" + name)
    next=path+ "\\" + name
    nextnameL=name.split(".")
    nextname=nextnameL[0]
    ListOfCompleteFilePaths.append(next)
    ListOfJustFileNames.append(nextname)
#print("DONE...")
print("full list...")
print(ListOfCompleteFilePaths)
print(ListOfJustFileNames)



# Read corpus of document into dataframe
# https://stackoverflow.com/questions/49088978/how-to-create-corpus-from-pandas-data-frame-to-operate-with-nltk
my_corpus=CategorizedPlaintextCorpusReader('C:/Users/rkrishnan/Documents/01 Personal/MS/IST 736/Week3/fedPapers/',r'.*', cat_pattern=r'*.txt') 
my_corpus.fileids() # <- I expect values from column ID
#my_corpus.categories() # <- I don't have a category defined in this case
my_corpus.words(fileids='dispt_fed_52.txt') # <- I expect values from column TITLE and BODY
my_corpus.sents(fileids=['dispt_fed_62.txt', 'dispt_fed_63.txt']) # <- I expect values from column TITLE and BODY

ListOffiles= pd.DataFrame(my_corpus.fileids(),columns=['file_nm'])

bow,doc=[],[]
for index in range(85):
    print(index,ListOffiles['file_nm'][index])
    doc.append(ListOffiles['file_nm'][index])
    my_list = my_corpus.words(fileids=ListOffiles['file_nm'][index])[:]
    bow.append(list(my_list))

##df= pd.DataFrame()
##list1 = list(range(10))
##list2 = list(range(10,20))
##df['list1'] = list1
##df['list2'] = list2
##print(df)
#
Federalist_DF=pd.DataFrame()
Federalist_DF['doc'] =doc
Federalist_DF['bow'] =bow

# Python3 program to convert a list 
# of integers into a single integer 
def convert(list): 
      
    # Converting integer list to string list 
    s = [str(i) for i in list] 
      
    # Join list items using join() 
    res = ("".join(s)) 
      
    return(res) 
  
# Driver code 
#list = ["aa", "bb", "cc"] 
#print(convert(list)) 
for index, paper in Federalist_DF.head(n=2).iterrows():
     print(index, paper)

# convert to lower case
Federalist_DF['bow'] = Federalist_DF['bow'].apply(lambda x: " ".join(x.lower() for x in x))
Federalist_DF['bow'].head()

#remove stopwords
stop = stopwords.words('english')
stop.append("the")
Federalist_DF['bow'] = Federalist_DF['bow'].apply(lambda x: " ".join(x for x in  x.split() if x not in stop))
Federalist_DF['bow'].head()



for index, paper in Federalist_DF.iterrows():
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(convert(paper))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(Federalist_DF['doc'][index])
    plt.show()


## CountVectorizers be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer

MyVect=CountVectorizer(input='filename')
## NOw I can vectorize using my list of complete paths to my files
X_DH=MyVect.fit_transform(ListOfCompleteFilePaths)

## NOw - what do we have?
##print(X_DH)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNames=MyVect.get_feature_names()
#print(ColumnNames3)

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_FP=pd.DataFrame(X_DH.toarray(),columns=ColumnNames)
print(CorpusDF_FP)
## Now update the row names
MyDict={}
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]

print("MY DICT:", MyDict)
        
CorpusDF_FP=CorpusDF_FP.rename(MyDict, axis="index")
print(CorpusDF_FP)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have


print(type(CorpusDF_FP))


## We have what we expected - a data frame.

# Convert DataFrame to matrix
MyMatrixFP = CorpusDF_FP.values
## Check it

print(type(MyMatrixFP))
print(MyMatrixFP)

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object.fit(MyMatrixFP)
# Get cluster assignment labels
labels = kmeans_object.labels_
#print(labels)
# Format results as a DataFrame
Myresults = pd.DataFrame([CorpusDF_FP.index,labels]).T
#print(Myresults)

#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults[0],Myresults[1],c=Myresults[1],s=250)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)


### Hmmm -  these are not great results
## This is because my dataset if not clean
## I still have stopwords
## I still have useless or small words < size 3

## Let's clean it up....
## Let's start with this: 

print(CorpusDF_FP)



## Let's remove our own stopwords that WE create

## Let's also remove all words of size 2 or smaller
## Finally, without using a stem package - 
## Let's combine columns with dog, dogs
## and with hike, hikes, hiking

## We need to "build" this in steps.
## First, I know that I need to be able to access
## the columns...

for name in ColumnNames:
    print(name)

## OK - that works...
## Now access the column by name

for name in ColumnNames:
    print(CorpusDF_FP[name])

## OK - can we "add" columns??
## lets test some things first
    
 
#name1="hikes"
#name2="hike"
#if(name1 == name2):
#    print("TRUE")
#else:
#    print("FALSE")
#
#name1=name1.rstrip("s")
#print(name1)
#if(name1 == name2):
#    print("TRUE")
#else:
#    print("FALSE")
#    
    #############################

## RE: https://docs.python.org/2.0/lib/module-string.html
## Now - let's put these ideas together
    ## note that strip() takes off the front
    ## rstrip() takes off the rear
## BEFORE

   

############################################
### Had a very odd 3-hour issue
### My code would not remove or see "and"
### Then I noticed that my for loop was
### "skipping" items that I thought were there
### WHy is this true?
### SOlution - always make a copy
### do not "do work" as you iterate - it
### messes up the index behind the scenes

###########################################
print("The initial column names:\n", ColumnNames)
print(type(ColumnNames))  ## This is a list
MyStops=["also", "and", "are", "you", "of", "let", "not", "the", "for", "why", "there", "one", "which","Federalist"]   


#df = pd.DataFrame(np.arange(12).reshape(3, 4),columns=['A', 'B', 'C', 'D'])
#df.drop(['B', 'C'], axis=1)
#df.drop(columns=['B', 'C'])
#df.drop([0, 1])

#from io import StringIO
#txt = """a, a, a, b, c, d
#1, 2, 3, 4, 5, 6
#7, 8, 9, 10, 11, 12"""
#
#df = pd.read_csv(StringIO(txt), skipinitialspace=True)
#print(df)
#print((df.filter(like='a')))
#
#df['a'] = df['a'] + df['a.1']+df['a.2']
#df=df.drop(['a.1','a.2'],axis=1)
#print(df)
#df=df.rename(columns={"c": "b"})
#print(df)
#df=df.groupby(level=0, axis=1).sum()
#pd.DataFrame(df.astype(bool).sum(axis=0)).T
#df['sum']=df.astype(bool).sum(axis=1)
#sum_col=len(df.columns)-1
#df.iloc[:,:sum_col].div(df["sum"], axis=0)
#
#i_df=pd.DataFrame((len(df)/df.astype(bool).sum(axis=0)).apply(lambda x: math.log(x))).T
#tf_idf=df*i_df.values


 ## MAKE COPIES!
CleanDF=CorpusDF_FP
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
        CleanDF=CleanDF.rename(columns={name: lem.lemmatize(name,"v")})
        ColNames.append(lem.lemmatize(name,"v"))
    else:
        ## I MUST add these new ColNames
        ColNames.append(name)
        

#print("END\n",CleanDF)             
print("The ending column names:\n", ColNames)

# Vectorize Term Frequencies begins here
ColNames_DF =pd.DataFrame()
ColNames_DF['ColNames']=ColNames
ColNames_DF['cnt']=1
ColNames_DF=ColNames_DF.groupby(['ColNames']).sum()
ColNames_DF=(ColNames_DF[ColNames_DF['cnt']>1])

CleanDF=CleanDF.groupby(level=0, axis=1).sum()


CleanDF_TF=CleanDF
CleanDF_TF['rowsum']=CleanDF_TF.astype(bool).sum(axis=1)
sum_col=len(CleanDF_TF.columns)-1
CleanDF_TF=CleanDF_TF.iloc[:,:sum_col].div(CleanDF_TF["rowsum"], axis=0)
CleanDF_TF=CleanDF_TF.drop(["rowsum"], axis=1)
FP_WordCount=CleanDF['rowsum']


# plot the word counts
#df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
#ax = df.plot.bar(x='lab', y='val', rot=0)
fig = plt.figure(figsize=(40,10))
ax = FP_WordCount.plot.bar(x='index', y='rowsum', rot=0)
#fig.add_subplot(111)
#bar = ax.bar(x=FP_WordCount.index.values, y=CleanDF.astype(bool).sum(axis=1), rot=0)
ax.set_title('Word Count on Federalist Papers')
ax.set_xlabel('Fedaralist Papers')
ax.set_ylabel('Count of words')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)

plt.colorbar(ax)



# Convert DataFrame to matrix
MyTFMatrixClean = CleanDF_TF.values


# Vectorize Term Frequencies ends here

IDF=pd.DataFrame((len(CleanDF_TF)/CleanDF_TF.astype(bool).sum(axis=0)).apply(lambda x: math.log(x))).T
CleanDF_TFIDF=CleanDF_TF*IDF.values
CleanDF_TFIDF['upon']

# Convert DataFrame to matrix
MyTFIDFMatrixClean = CleanDF_TFIDF.values

# Vectorize TFIDF Frequencies begins here


 
# Vectorize TFIDF Frequencies ends here

#for name1 in ColNames:
#    for name2 in ColNames:
#        if(name1 == name2):
#            print("skip")
#        elif(name1.rstrip("e") in name2):  ## this is good for plurals
#            ## like dog and dogs, but not for hike and hiking
#            ## so I will strip an "e" if there is one...
#            print("combining: ", name1, name2)
#            print(CorpusDF_FP[name1])
#            print(CorpusDF_FP[name2])
#            print(CorpusDF_FP[name1] + CorpusDF_FP[name2])
#            
#            ## Think about how to test this!
#            ## at first, you can do this:
#            ## NEW=name1+name2
#            ## CleanDF[NEW]=CleanDF[name1] + CleanDF[name2]
#            ## Then, before dropping any columns - print
#            ## the columns and their sum to check it. 
#            
#            CleanDF[name1] = CleanDF[name1] + CleanDF[name2]
#            
#            ### Later and once everything is tested - you
#            ## will include this next line of code. 
#            ## While I tested everyting, I had this commented out
#            ###   "******
#            CleanDF=CleanDF.drop([name2], axis=1)
#        
print(CleanDF.columns.values)

## Confirm that your column summing is working!

print(CleanDF["zealous"])
#print(CleanDF["dogs"])
#print(CleanDF["dogdogs"])  ## this should be the sum

## AFTER
print(CleanDF)

## NOW - let's try k means again....
############################## k means ########################

# Convert DataFrame to matrix
MyMatrixClean = CleanDF.values
## Check it
print(type(MyMatrixClean))
print(MyMatrixClean)

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object2 = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object2.fit(MyMatrixClean)
# Get cluster assignment labels
labels2 = kmeans_object2.labels_
print("k-means with k = 3\n", labels2)
# Format results as a DataFrame
Myresults2 = pd.DataFrame([CleanDF.index,labels2]).T
Myresults2.columns=['Paper','Cluster']
print("k means RESULTS\n", Myresults2)


#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults2['Paper'],Myresults2['Cluster'],c=Myresults2['Cluster'],s=250)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)

Myresults3=pd.DataFrame(Myresults2)
Myresults3['size'] = Myresults3.Paper.str.replace(r'(^.*Hamilton.*$)', "250").str.replace(r'(^.*Madison.*$)', "125").str.replace(r'(^.*dispt.*$)', "375").str.replace(r'(^.*[a-z].*$)', "50")
Myresults3.index=CleanDF.index
Myresults3['upon']=CleanDF['upon']
Myresults3=Myresults3.reset_index()
Myresults3=Myresults3.drop(['index'], axis=1)

#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults3['Paper'],Myresults3['upon'],c=Myresults3['Cluster'],s= pd.to_numeric(Myresults3['size']))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('token count for "upon"')
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
labels3 = kmeans_tf_object.labels_
print("k-means with k = 2\n", labels3)
# Format results as a DataFrame
Myresults4 = pd.DataFrame([CleanDF_TF.index,labels3]).T
Myresults4.columns=['Paper','Cluster']
print("k means RESULTS\n", Myresults4)


#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults4['Paper'],Myresults4['Cluster'],c=Myresults4['Cluster'],s=250)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)

Myresults5=pd.DataFrame(Myresults4)
Myresults5['size'] = Myresults5.Paper.str.replace(r'(^.*Hamilton.*$)', "250").str.replace(r'(^.*Madison.*$)', "125").str.replace(r'(^.*dispt.*$)', "375").str.replace(r'(^.*[a-z].*$)', "50")
Myresults5.index=CleanDF_TF.index
Myresults5['upon']=CleanDF_TF['upon']
Myresults5=Myresults5.reset_index()
Myresults5=Myresults5.drop(['index'], axis=1)

#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults5['Paper'],Myresults5['upon'],c=Myresults5['Cluster'],s= pd.to_numeric(Myresults5['size']))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('token count for "upon"')
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
labels4 = kmeans_tfidf_object.labels_
print("k-means with k = 2\n", labels4)
# Format results as a DataFrame
Myresults6 = pd.DataFrame([CleanDF_TFIDF.index,labels4]).T
Myresults6.columns=['Paper','Cluster']
print("k means RESULTS\n", Myresults6)


#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults6['Paper'],Myresults6['Cluster'],c=Myresults6['Cluster'],s=250)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('Clusters')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.colorbar(scatter)

Myresults7=pd.DataFrame(Myresults6)
Myresults7['size'] = Myresults7.Paper.str.replace(r'(^.*Hamilton.*$)', "250").str.replace(r'(^.*Madison.*$)', "125").str.replace(r'(^.*dispt.*$)', "375").str.replace(r'(^.*[a-z].*$)', "50")
Myresults7.index=CleanDF_TFIDF.index
Myresults7['upon']=CleanDF_TFIDF['upon']
Myresults7=Myresults7.reset_index()
Myresults7=Myresults7.drop(['index'], axis=1)

#Plot the clusters obtained using k means
fig = plt.figure(figsize=(40,10))
ax = fig.add_subplot(111)
colors = {0:'red', 1:'blue'}
scatter = ax.scatter(Myresults7['Paper'],Myresults7['upon'],c=Myresults7['Cluster'],s= pd.to_numeric(Myresults7['size']))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Fedaralist Paper Numbers')
ax.set_ylabel('token count for "upon"')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)

plt.colorbar(scatter)