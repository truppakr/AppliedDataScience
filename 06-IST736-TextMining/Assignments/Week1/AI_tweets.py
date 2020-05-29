# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:10:55 2019

@author: rkrishnan
"""

# Import the Twython class
from twython import Twython  
import json

# Load credentials from json file
with open("twitter_credentials.json", "r") as file:  
    creds = json.load(file)

# Instantiate an object
python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

# Create our query
query = {'q': 'artificial intelligence ',  
        'result_type': 'recent',
        'count': 100,
        'lang': 'en',
        }


import pandas as pd

# Search tweets
dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}  
for status in python_tweets.search(**query)['statuses']:  
    dict_['user'].append(status['user']['screen_name'])
    dict_['date'].append(status['created_at'])
    dict_['text'].append(status['text'])
    dict_['favorite_count'].append(status['favorite_count'])

# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)  
df.sort_values(by='favorite_count', inplace=True, ascending=False)  
df.head(5)  
df['text'].head()

# Save the tweets object to file
with open("ai_tweets.json", "w") as file:  
    json.dump(dict_, file)

file.close()
    


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import re
df['new_text']=df['text']
for eachrow in df.index:
    df['new_text'][eachrow] = re.sub(r"http\S+", "", df['text'][eachrow])
    df['new_text'][eachrow] = re.sub('RT @[^\s]+','', df['new_text'][eachrow])

df['new_text'].to_csv(r'C:\Users\rkrishnan\Documents\01 Personal\MS\IST 736\Week1\ai_tweets.txt', header=None, index=None, sep=' ', mode='w')

sid = SentimentIntensityAnalyzer()
for sentence in df['text']:
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()
