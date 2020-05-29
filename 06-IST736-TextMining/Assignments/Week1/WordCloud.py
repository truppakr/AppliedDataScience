# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:52:14 2019

@author: rkrishnan
"""

# word cloud
# reference: https://www.datacamp.com/community/tutorials/wordcloud-python

# if you haven't installed the wordcloud package, try pip install outside this script

# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#% matplotlib inline


# Start with one review:
text = open("C:/Users/rkrishnan/Documents/01 Personal/MS/IST 736/Week1/Injuries.txt").read()

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