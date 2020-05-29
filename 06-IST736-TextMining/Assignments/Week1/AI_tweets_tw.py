# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:56:02 2019

@author: rkrishnan
"""
import tweepy as tw
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pandas as pd
import os

cwd = os.getcwd()

# Load credentials from json file
with open("twitter_credentials.json", "r") as file:  
    creds = json.load(file)

# Instantiate an object
auth = tw.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
api = tw.API(auth, wait_on_rate_limit=True)


search_str = "artificial+intelligence -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=search_str,
                   lang="en",
                   since='2018-04-23',tweet_mode='extended').items(1000)

all_tweets = [tweet.full_text for tweet in tweets]
all_tweets[:5]

#Convert list of tuples to dataframe and set column names and indexes
tweet_df = pd.DataFrame(all_tweets, columns = ['tweet']) 

tweet_df['new_tweet']=tweet_df['tweet']

for eachrow in tweet_df.index:
    tweet_df['new_tweet'][eachrow] = re.sub(r"http\S+", "", tweet_df['new_tweet'][eachrow])
    tweet_df['new_tweet'][eachrow] = re.sub('RT @[^\s]+','', tweet_df['new_tweet'][eachrow])
    tweet_df['new_tweet'][eachrow] = re.sub('#[^\s]+','', tweet_df['new_tweet'][eachrow])
    tweet_df['new_tweet'][eachrow] = re.sub('@[^\s]+','', tweet_df['new_tweet'][eachrow])
    tweet_df['new_tweet'][eachrow] = ''.join([c for c in tweet_df['new_tweet'][eachrow] if ord(c) < 128])

tweet_df['new_tweet'].to_csv(r'ai_tweets_tw.txt', header=['new_tweet'], index=None, sep=' ', mode='w')

ai_tweets = pd.read_csv('ai_tweets_tw.txt' )

sid = SentimentIntensityAnalyzer()
for sentence in ai_tweets['new_tweet']:
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()

