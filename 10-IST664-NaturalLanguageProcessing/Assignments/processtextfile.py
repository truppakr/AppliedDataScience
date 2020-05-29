'''
	This program reads and processes text from a file.
	To change the file, change the filepath variable
	It also reads a custom stopword file called Smart.English.stop
	The program gets the raw text, tokenizes and lowercases the tokens.
	It puts the tokens in a frequency distribution and displays the 30 most frequent.
'''
# open python and nltk packages needed for processing
import nltk
import re
from nltk.collocations import *
import pandas as pd
# put the full path to the file here (or can use relative path from the directory of the program)
#filepath = '/Users/njmccrac/NLPfall2016/labs/LabExamplesWeek4/CrimeAndPunishment.txt'
#filepath = 'H:\NLPclass\LabExamplesWeek4\CrimeAndPunishment.txt'
#filepath = 'C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST664\\Week3\\CrimeAndPunishment.txt'
filepath = 'C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST664\\Week3\\desert.txt'

def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

# open the file, read the text and close it
f = open(filepath, 'r')
filetext = f.read()
f.close()

len(filetext)
# tokenize by the regular word tokenizer
filetokens = nltk.word_tokenize(filetext)
len(filetokens)
# choose to treat upper and lower case the same
#    by putting all tokens in lower case
filewords = [w.lower() for w in filetokens]


# display the first words
print ("Display first 50 words from file:")
print (filewords[:50])

# read a stop word file
fstop = open('C:\\Users\\rkrishnan\\Documents\\01 Personal\\MS\\IST664\\Week3\\Smart.English.stop', 'r')
stoptext = fstop.read()
fstop.close()

stopwords = nltk.word_tokenize(stoptext)
print ("Display first 50 Stopwords:")
print (stopwords[:50])

# setup to process bigrams

bigram_measures = nltk.collocations.BigramAssocMeasures()
     
finder = BigramCollocationFinder.from_words(filewords)
# choose to use both the non-alpha word filter and a stopwords filter
finder.apply_word_filter(alpha_filter)
finder.apply_word_filter(lambda w: w in stopwords)

# score by frequency and display the top 50 bigrams
scored = finder.score_ngrams(bigram_measures.raw_freq)
print ()
print ("Bigrams from file with top 50 frequencies")
for item in scored[:20]:
        print (item)

# score by PMI and display the top 50 bigrams
# only use frequently occurring words in mutual information
finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.pmi)

print ("\nBigrams from file with top 50 mutual information scores")
for item in scored[:20]:
        print (item)


porter=nltk.PorterStemmer()
lancaster=nltk.LancasterStemmer()


# stem file words using porter and lancaster stemmer 

porter_stem_words=[porter.stem(w) for w in filewords]
print(porter_stem_words[100:200])

lancaster_stem_words=[lancaster.stem(w) for w in filewords]
print(lancaster_stem_words[100:200])


def compare_stems(filewords,porter_stem_words,lancaster_stem_words):
    result=[]
    result_dict={}
    for idx,w in enumerate(filewords):
        if w!=porter_stem_words[idx] and w!=lancaster_stem_words[idx] and porter_stem_words[idx]==lancaster_stem_words[idx]:
            result.append("Stemming is same in porter and lancaster")
        elif w!=porter_stem_words[idx] and w!=lancaster_stem_words[idx] and porter_stem_words[idx]!=lancaster_stem_words[idx]:
            result.append("Stemming is different in porter and lancaster")
        elif w==porter_stem_words[idx] and w==lancaster_stem_words[idx]:
            result.append("No stemming on both porter and lancaster")
        elif w==porter_stem_words[idx] and w!=lancaster_stem_words[idx]:
            result.append("Stemmed on lancaster but not on porter")
        elif w!=porter_stem_words[idx] and w==lancaster_stem_words[idx]:
            result.append("Stemmed on porter but not on lancaster")
        else:
            result.append("No result")
    result_dict={'filewords':filewords,'porter_stem_words':porter_stem_words,'lancaster_stem_words':lancaster_stem_words,'result':result}
    return result,result_dict
    
    
stem_result,result_dict =compare_stems(filewords[100:200],porter_stem_words[100:200],lancaster_stem_words[100:200])

result_df=pd.DataFrame(result_dict)

print(result_df)