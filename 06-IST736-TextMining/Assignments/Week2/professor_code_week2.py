# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 06:52:49 2019

@author: rkrishnan
"""

###########
## Class - remember to change the path
## Type this in BY HAND - copy/paste can not only create hidden chars and so errors
## but is also not as good practice.
## Run and Test AS YOU TYPE - do not just copy/paste/run
## Feel free to comment out or add in print statements
##########################################################
# -*- coding: utf-8 -*-
"""


@author: profa
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
from nltk.tokenize import sent_tokenize, word_tokenize
import os

#MyCorpusReader = nltk.corpus.PlaintextCorpusReader('CorpusHikeDog_Small', '.*\.txt')
#print(MyCorpusReader)
#print(MyCorpusReader.raw())
###BUT.. the above is not what I want. So I will write my own methods...

## I know that using"filename" with CountVectorizer requires a LIST
## of the file names. So I need to build one....

path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\CorpusHikeDog_Small"
print("calling os...")
## Print the files in this location to make sure I am where I want to be
print(os.listdir(path))
## Save the list
FileNameList=os.listdir(path)
## check the TYPE
print(type(FileNameList))
print(FileNameList)

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

## CountVectorizers be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer


MyVect3=CountVectorizer(input='filename')
## NOw I can vectorize using my list of complete paths to my files
X_DH=MyVect3.fit_transform(ListOfCompleteFilePaths)

## NOw - what do we have?
##print(X_DH)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNames3=MyVect3.get_feature_names()
#print(ColumnNames3)

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_DogHike=pd.DataFrame(X_DH.toarray(),columns=ColumnNames3)
print(CorpusDF_DogHike)
## Now update the row names
MyDict={}
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]

print("MY DICT:", MyDict)
        
CorpusDF_DogHike=CorpusDF_DogHike.rename(MyDict, axis="index")
print(CorpusDF_DogHike)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have


print(type(CorpusDF_DogHike))


## We have what we expected - a data frame.

# Convert DataFrame to matrix
MyMatrixDogHike = CorpusDF_DogHike.values
## Check it

print(type(MyMatrixDogHike))
print(MyMatrixDogHike)

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object = sklearn.cluster.KMeans(n_clusters=3)
#print(kmeans_object)
kmeans_object.fit(MyMatrixDogHike)
# Get cluster assignment labels
labels = kmeans_object.labels_
#print(labels)
# Format results as a DataFrame
Myresults = pd.DataFrame([CorpusDF_DogHike.index,labels]).T
#print(Myresults)

### Hmmm -  these are not great results
## This is because my dataset if not clean
## I still have stopwords
## I still have useless or small words < size 3

## Let's clean it up....
## Let's start with this: 

print(CorpusDF_DogHike)



## Let's remove our own stopwords that WE create

## Let's also remove all words of size 2 or smaller
## Finally, without using a stem package - 
## Let's combine columns with dog, dogs
## and with hike, hikes, hiking

## We need to "build" this in steps.
## First, I know that I need to be able to access
## the columns...

for name in ColumnNames3:
    print(name)

## OK - that works...
## Now access the column by name

for name in ColumnNames3:
    print(CorpusDF_DogHike[name])

## OK - can we "add" columns??
## lets test some things first
    
 
name1="hikes"
name2="hike"
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")

name1=name1.rstrip("s")
print(name1)
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")
    
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
print("The initial column names:\n", ColumnNames3)
print(type(ColumnNames3))  ## This is a list
MyStops=["also", "and", "are", "you", "of", "let", "not", "the", "for", "why", "there", "one", "which"]   

 ## MAKE COPIES!
CleanDF=CorpusDF_DogHike
print("START\n",CleanDF)
## Build a new columns list
ColNames=[]

for name in ColumnNames3:
    #print("FFFFFFFF",name)
    if ((name in MyStops) or (len(name)<3)):
        #print("Dropping: ", name)
        CleanDF=CleanDF.drop([name], axis=1)
        #print(CleanDF)
    else:
        ## I MUST add these new ColNames
        ColNames.append(name)
        

#print("END\n",CleanDF)             
print("The ending column names:\n", ColNames)


for name1 in ColNames:
    for name2 in ColNames:
        if(name1 == name2):
            print("skip")
        elif(name1.rstrip("e") in name2):  ## this is good for plurals
            ## like dog and dogs, but not for hike and hiking
            ## so I will strip an "e" if there is one...
            print("combining: ", name1, name2)
            print(CorpusDF_DogHike[name1])
            print(CorpusDF_DogHike[name2])
            print(CorpusDF_DogHike[name1] + CorpusDF_DogHike[name2])
            
            ## Think about how to test this!
            ## at first, you can do this:
            ## NEW=name1+name2
            ## CleanDF[NEW]=CleanDF[name1] + CleanDF[name2]
            ## Then, before dropping any columns - print
            ## the columns and their sum to check it. 
            
            CleanDF[name1] = CleanDF[name1] + CleanDF[name2]
            
            ### Later and once everything is tested - you
            ## will include this next line of code. 
            ## While I tested everyting, I had this commented out
            ###   "******
            CleanDF=CleanDF.drop([name2], axis=1)
        
print(CleanDF.columns.values)

## Confirm that your column summing is working!

print(CleanDF["dog"])
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
kmeans_object2 = sklearn.cluster.KMeans(n_clusters=3)
#print(kmeans_object)
kmeans_object2.fit(MyMatrixClean)
# Get cluster assignment labels
labels2 = kmeans_object2.labels_
print("k-means with k = 3\n", labels2)
# Format results as a DataFrame
Myresults2 = pd.DataFrame([CleanDF.index,labels2]).T
print("k means RESULTS\n", Myresults2)

################# k means with k = 2 #####################


# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object3 = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object3.fit(MyMatrixClean)
# Get cluster assignment labels
labels3 = kmeans_object3.labels_
print("K means with k = 2\n", labels3)
# Format results as a DataFrame
Myresults3 = pd.DataFrame([CleanDF.index,labels3]).T
print("k means RESULTS\n", Myresults3)