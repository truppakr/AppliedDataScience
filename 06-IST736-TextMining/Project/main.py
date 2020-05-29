import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import os
from config import path
from matplotlib_venn import venn2
import matplotlib.pyplot as plt,mpld3
import re
#from wordcloud import WordCloud
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.ensemble import ExtraTreesClassifier
import operator
from nltk.probability import FreqDist


from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
#from sklearn.externals import joblib
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram

from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from time import time


# Join Ted Main and Transcripts ###############################################
os.chdir(path)

# Read ted files to a dataframe
ted_main = pd.read_csv(path + "\\ted_main.csv") 
ted_transcript = pd.read_csv(path + "\\transcripts.csv") 

# Inner join based on URL
ted_join = pd.merge(ted_main, ted_transcript, on='url', how='inner')

def print_venn(a, b, title, labels):
  out = venn2([set(a), set(b)], set_labels=labels)
  for text in out.subset_labels:
    text.set_fontsize(30)
  plt.title(title)#, fontsize=30)
  plt.show()  
  
plt.rcParams.update({'font.size': 20})
print_venn(ted_main['url'], ted_transcript['url'], "Join Ted Talk Datasets", ('ted_main', 'transcripts'))

# Drop duplicates and rows which contain missing values
ted_clean = ted_join.drop_duplicates()
ted_clean = ted_clean.dropna()

# Remove all non ted talks
ted_clean = ted_clean[ted_clean['event'].str[0:3] == 'TED']

# Remove ted talks focused on a local community (concentrates on local voices 
# TEDx stands for independently organized TED events)
ted_clean = ted_clean[ted_clean['event'].str[0:4] != 'TEDx']

# Make dates readable and Check year (Ted established in 1984)
ted_clean['film_date'] = ted_clean['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
year = pd.DataFrame(ted_clean['film_date'].apply(lambda x: x.year < 1984))
len(year[year['film_date'] == True])
ted_clean['published_date'] = ted_clean['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
year = pd.DataFrame(ted_clean['published_date'].apply(lambda x: x.year < 1984))
len(year[year['published_date'] == True])

# Remove ted talks filmed ealier than 2010
year = pd.DataFrame(ted_clean['film_date'].apply(lambda x: x.year >= 2010))
ted_clean['within_decade'] =  year
ted_clean = ted_clean[ted_clean['within_decade'] == True]

# Create Labels ###############################################################
# Create sentiment lexicon
ted_clean['ratings'] = ted_clean['ratings'].str.replace("'",'"')
list(set([item for ratings in  ted_clean['ratings'] for item in pd.read_json(ratings)['name']]))
sentiment = {'positive': ['Beautiful', 'Funny', 'Persuasive', 'Courageous', 'Inspiring', 'Fascinating', 'Ingenious','Informative', 'Jaw-dropping'],
             'neutral':  ['OK'],
             'negative': ['Unconvincing', 'Longwinded', 'Obnoxious', 'Confusing']}

# Sum sentiment ratings for each transcript (plus 1 to avoid dividing by 0 later)
ratings  = ted_clean.ratings.apply(lambda x: pd.Series(pd.read_json(x)['count'].values,index=pd.read_json(x)['name']))
positive = ratings.loc[:,sentiment['positive']].sum(axis=1)+1
negative = ratings.loc[:,sentiment['negative']].sum(axis=1)+1

# Calculate popularity ratio
popular_ratio = positive/negative

# Get minimum outlier
sns.boxplot(popular_ratio)
bp = plt.boxplot(popular_ratio)
outliers = [item.get_ydata() for item in bp['fliers']]
outliers[0].min()

# Get right whisker
whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]
whiskers[1].max()

# Cap outliers
popular_ratio[popular_ratio >= outliers[0].min()] = whiskers[1].max()
sns.boxplot(popular_ratio)
popular_ratio.describe()

# Define popular scores
labels = []
lower = np.percentile(popular_ratio, 25)
upper = np.percentile(popular_ratio, 75)
for score in popular_ratio:
    label = "nominal"
    if score >= upper: label = "popular"
    if score <= lower: label = "unpopular" 
    labels.append(label)

# Add labels and remove rows that contain nominal labels
ted_clean['labels'] = labels
ted_clean = ted_clean[ted_clean['labels'] != 'nominal']
ted_clean['labels'].value_counts().plot('bar', title='Ted Talk Labels')

# Save to CSV
ted_clean.to_csv('ted_clean.tsv', encoding='utf-8', index=False, sep='\t')

# Feature Generation ##########################################################
def get_important_features(data, labels):  
    # Reduce Vocabulary Size with Tree Base Feature Selection    
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(data)
    
    forest = ExtraTreesClassifier(n_estimators=100)
    forest = forest.fit(x, labels)
    importances = dict(zip(vectorizer.get_feature_names(), forest.feature_importances_))
    threshold = np.percentile(np.array(list(importances.values())), 99)
    
    # Sort and Filter Features with Values less than the Threshold
    importances = {key:value for key, value in importances.items() if value > threshold}
    return dict(sorted(importances.items(), key=operator.itemgetter(1),reverse=True))

important_features = get_important_features(ted_clean['transcript'], ted_clean['labels'])
fd = FreqDist(important_features)
fd.plot(100, cumulative=False)

def process_unigrams(data, threshold=0.0):
    return [key for key, value in data.items() if value > threshold]

unigrams = process_unigrams(important_features)

def alpha_filter(word):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(word)):
        return True
    else:
        return False
    
def collocate_bigrams(transcripts):
    bigram_measures = BigramAssocMeasures()
    # tokenize transcripts
    tokens = [token for text in transcripts for token in wordpunct_tokenize(text)]
    # must supply unfiltered tokens because may incorrectly put together two words 
    # that might have been separated by non-alphabetic token or a stop word token.
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_word_filter(alpha_filter) 
    finder.apply_freq_filter(5)
    return finder.score_ngrams(bigram_measures.pmi)
        
popular_mi  = collocate_bigrams(ted_clean[ted_clean['labels'] == 'popular']['transcript'])
unpopular_mi = collocate_bigrams(ted_clean[ted_clean['labels'] == 'unpopular']['transcript'])

def process_bigrams(data, threshold=1):
    bigrams = [x[0] for x in data if x[1] > threshold]
    return [" ".join(pair).lower() for pair in bigrams if len(pair[0]) > 1 and len(pair[1]) > 1]
    
popular_bigrams = process_bigrams(popular_mi, 10)
unpopular_bigrams = process_bigrams(unpopular_mi, 10)

print_venn(popular_bigrams, unpopular_bigrams, "Ted Talks Bigrams", ['Popular', 'Unpopular'])
popular_unique_bigrams = list(set(popular_bigrams) - set(unpopular_bigrams))
unpopular_unique_bigrams = list(set(unpopular_bigrams) - set(popular_bigrams))

bag_of_ngrams = popular_unique_bigrams + unpopular_unique_bigrams + unigrams

#def draw_wordcloud(data, title, color='white'):
#    words = ' '.join(data)
#    wordcloud = WordCloud(background_color=color, 
#                          width=2500, 
#                          height=2000).generate(words)
#    plt.figure(1,figsize=(13, 13))
#    plt.imshow(wordcloud)
#    plt.title(title, fontsize=20)
#    plt.axis('off')
#    plt.show()
    
#draw_wordcloud(popular_unique_bigrams, 'Popular Ted Talks', color='white')
#draw_wordcloud(unpopular_unique_bigrams, 'Unpopular Ted Talks', color='white')

# Vectorize Ted Talks #########################################################
def vectorize_text(data, vectorizer):
    x = vectorizer.fit_transform(data)
    return pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

# boolean vectorizer
bool_vectorizer = CountVectorizer(vocabulary=bag_of_ngrams, encoding='latin-1', ngram_range=(1,2), binary=True)
bool_df = vectorize_text(ted_clean['transcript'], bool_vectorizer)
bool_df['labels'] = list(ted_clean['labels'])

# term frequency vectorizer
count_vectorizer = CountVectorizer(vocabulary=bag_of_ngrams, encoding='latin-1', ngram_range=(1,2), binary=False)
count_df = vectorize_text(ted_clean['transcript'], count_vectorizer)
count_df['labels'] = list(ted_clean['labels'])
vectorizer = make_pipeline(count_vectorizer, TfidfTransformer())

# tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(vocabulary=bag_of_ngrams, encoding='latin-1', ngram_range=(1,2), use_idf=True)
tfidf_df = vectorize_text(ted_clean['transcript'], tfidf_vectorizer)
tfidf_df['labels'] = list(ted_clean['labels'])

# Save to CSV
path = ".\\vectorized_data\\"
bool_df.to_csv(path + 'bool_ted.csv', encoding='utf-8', index=False)
count_df.to_csv(path + 'count_ted.csv', encoding='utf-8', index=False)
tfidf_df.to_csv(path + 'tfidf_ted.csv', encoding='utf-8', index=False)



#############################################################
########### Clustering Analysis #############################
#############################################################  



def get_distance(data,method):
    ## Get the euclidean dist between TED talks
    #Using sklearn
    dist=[]
    if method=="euclidean":
        dist=euclidean_distances(data.iloc[:,0:(data.shape[1]-1)])
    elif method=="cosine":
        dist=1- cosine_similarity(data.iloc[:,0:(data.shape[1]-1)])
    else:
        print("Function accepts euclidean and cosine distance measures.")
    return dist

def get_clusters(distance_matrix,K=3):
    #from sklearn.cluster import KMeans
    km = KMeans(init='k-means++', max_iter=10000, n_init=1,
                    verbose=0, n_clusters=K)
    km.fit(distance_matrix)
    clusters=km.labels_.tolist()
    return clusters

def get_cluster_frm_dist(data,method,K=3):
    
    def get_distance(data,method):
        ## Get the euclidean dist between TED talks
        #Using sklearn
        
        dist=[]
        if method=="euclidean":
            dist=euclidean_distances(data.iloc[:,0:(data.shape[1]-1)])
        elif method=="cosine":
            dist=1- cosine_similarity(data.iloc[:,0:(data.shape[1]-1)])
        else:
            print("Function accepts euclidean and cosine distance measures.")
        return dist

    def get_clusters(distance_matrix,K=3):
        #from sklearn.cluster import KMeans
        km = KMeans(init='k-means++', max_iter=10000, n_init=1,
                        verbose=0, n_clusters=K)
        km.fit(distance_matrix)
        #uncomment the below to save your model 
        #since I've already run my model I am loading from the pickle
        #joblib.dump(km,  'doc_cluster.pkl')
        #km = joblib.load('doc_cluster.pkl')
        clusters=km.labels_.tolist()
        #sort cluster centers by proximity to centroid
        order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
        return [clusters,order_centroids]
    
    return get_clusters(get_distance(data,method),K)

def get_top_terms_per_cluster(data,vectorizer,cluster,centroid,num_clusters,label='labels'):

    ted_cl = { 'talk': list(data.index),'cluster': cluster, 'labels': list(data[label])}
    frame = pd.DataFrame(ted_cl, index = [cluster] , columns = ['talk','cluster', 'labels'])
    frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to K)
    grouped = frame['talk'].groupby(frame['cluster']) #groupby cluster for aggregation purposes
    grouped.mean() #average rank (1 to K) per cluster
    terms = vectorizer.get_feature_names()
#    vocab_frame = pd.DataFrame({'words': list(count_df.columns)})
#    print ('There are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    print("Top terms per cluster:")
    print()
    for i in range(num_clusters):
        print("Cluster %d:" % i, end='')
        for ind in centroid[i, :15]:
            print(' %s' % terms[ind], end='')
        print()


def plot_clusters(distance_matrix,label_with_color,cluster_colors,cluster_names,file_nm):
    ## Visualizing Distances
    ##An option for visualizing distances is to assign a point in a plane
    ## to each text such that the distance between points is proportional 
    ## to the pairwise euclidean or cosine distances.
    ## This type of visualization is called multidimensional scaling (MDS) 
    ## in scikit-learn (and R  -  mdscale).
    cluster_df=label_with_color
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    ## "precomputed" means we will give the dist (as cosine sim)
    pos = mds.fit_transform(distance_matrix)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys,c=list(cluster_df.loc[:,'color']))
    plt.show()
    
    
#    ##PLotting the relative distances in 3D
#    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
#    pos = mds.fit_transform(distance_matrix)
#    #fig, ax = plt.subplots(figsize=(17, 9)) # set size
#    fig = plt.figure(figsize=(17, 9))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],c=list(cluster_df.loc[:,'color']))
#    plt.show()
#    
    
    #
    ##some ipython magic to show the matplotlib plots inline
    ##%matplotlib inline 
    #from IPython import get_ipython
    #get_ipython().run_line_magic('matplotlib', 'inline')
    #
    ##create data frame that has the result of the MDS plus the cluster numbers and tags
    #df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, ted_tags=(ted_clean['labels']))) 
    ##group by cluster
    #groups = df.groupby('label')
    ## set up plot
    #fig, ax = plt.subplots(figsize=(17, 9)) # set size
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #
    ##iterate through groups to layer the plot
    ##note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    #for name, group in groups:
    #    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, 
    #            label=cluster_names[name], color=cluster_colors[name], 
    #            mec='none')
    #    ax.set_aspect('auto')
    #    ax.tick_params(\
    #        axis= 'x',          # changes apply to the x-axis
    #        which='both',      # both major and minor ticks are affected
    #        bottom='off',      # ticks along the bottom edge are off
    #        top='off',         # ticks along the top edge are off
    #        labelbottom='off')
    #    ax.tick_params(\
    #        axis= 'y',         # changes apply to the y-axis
    #        which='both',      # both major and minor ticks are affected
    #        left='off',      # ticks along the bottom edge are off
    #        top='off',         # ticks along the top edge are off
    #        labelleft='off')
    #    
    #ax.legend(numpoints=1)  #show legend with only 1 point
    ##add label in x,y position with the label as the film title
    #for i in range(len(df)):
    #    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['labels'], size=8)  
    #plt.show() #show the plot
    
    
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
    df = pd.DataFrame(dict(x=xs, y=ys, label=cluster_df['cluster'], ted=(cluster_df['tags']),marker=cluster_df['marker'])) 
    #group by cluster
    groups = df.groupby(['label','marker'])
    
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
    fig, ax = plt.subplots(figsize=(25,10)) #set plot size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        if name[1]=='x':
            points = ax.plot(group.x, group.y, marker='^', linestyle='', ms=10, 
                             label= cluster_names[name[0]], mec='none', 
                             color=cluster_colors[name[0]])
        else:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, 
                 label= cluster_names[name[0]], mec='none', 
                 color=cluster_colors[name[0]])
        
        ax.set_aspect('auto')
        labels = [i for i in group.ted]
        
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
    
    f = open((file_nm +".html"),'w')
    
    f.write(html)
    f.close()
    
    ######### Alternative - this works
    ## Good tutorial http://brandonrose.org/clustering
    #from scipy.cluster.hierarchy import ward, dendrogram
    linkage_matrix = ward(count_cosdist) #define the linkage_matrix using ward clustering pre-computed distances
    
    fig, ax = plt.subplots(figsize=(20, 50)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=list(ted_clean.index));
    
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
    plt.tight_layout() #show plot with tight layout
    
    #uncomment below to save figure
    plt.savefig((file_nm +".png"), dpi=200) #save figure as ward_clusters
    

#Measure of distance that takes into account the
num_clusters = 11

# cluster dataframe with name and color
cluster_df = { 'cluster':[0,1,2,3,4,5,6,7,8,9,10],
              'color': ['#a50026','#d73027', '#f46d43','#fdae61','#fee090', '#ffffbf','#e0f3f8','#abd9e9', '#74add1','#4575b4', '#313695'],
              'name':['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster 8','Cluster 9','Cluster 10','Cluster 11']}

#set up colors per clusters using a dict
cluster_colors = {0: '#a50026', 
                 1: '#d73027',
                 2: '#f46d43',
                 3: '#fdae61', 
                 4: '#fee090',
                 5: '#ffffbf',
                 6: '#e0f3f8', 
                 7: '#abd9e9',
                 8: '#74add1',
                 9: '#4575b4', 
                 10: '#313695'
                }

#set up cluster names using a dict
cluster_names = {0: 'Cluster 1', 
                 1: 'Cluster 2',
                 2: 'Cluster 3',
                 3: 'Cluster 4', 
                 4: 'Cluster 5',
                 5: 'Cluster 6',
                 6: 'Cluster 7', 
                 7: 'Cluster 8',
                 8: 'Cluster 9',
                 9: 'Cluster 10', 
                 10: 'Cluster 11'
                }



bool_dist = get_distance(bool_df,"euclidean")
count_dist = get_distance(count_df,"euclidean")
tfidf_dist = get_distance(tfidf_df,"euclidean")

print(np.round(bool_dist,0))  #

#length of the document: called cosine similarity
bool_cosdist = get_distance(bool_df,"cosine")
count_cosdist = get_distance(count_df,"cosine")
tfidf_cosdist = get_distance(tfidf_df,"cosine")
print(np.round(count_cosdist,3))  #cos dist should be .02

# Get the cluster and centroid array from the function
bool_eu_clust,bool_eu_centroids=get_cluster_frm_dist(bool_df,"euclidean",num_clusters)
count_eu_clust,count_eu_centroids=get_cluster_frm_dist(count_df,"euclidean",num_clusters)
tfidf_eu_clust,tfidf_eu_centroids=get_cluster_frm_dist(tfidf_df,"euclidean",num_clusters)

bool_cos_clust,bool_cos_centroids=get_cluster_frm_dist(bool_df,"cosine",num_clusters)
count_cos_clust,count_cos_centroids=get_cluster_frm_dist(count_df,"cosine",num_clusters)
tfidf_cos_clust,tfidf_cos_centroids=get_cluster_frm_dist(tfidf_df,"cosine",num_clusters)

# Print tpe terms per cluster
#get_top_terms_per_cluster(ted_clean,bool_vectorizer,bool_eu_clust,bool_eu_centroids,num_clusters,label='labels')
#bool_eu_clust_df=pd.DataFrame(bool_eu_clust,columns=['cluster']).merge(pd.DataFrame.from_dict(cluster_df), on=['cluster'], how='left')
#bool_eu_clust_df=bool_eu_clust_df.merge(pd.DataFrame(list(ted_clean['tags']),columns=['tags']),left_index=True, right_index=True)
#plot_clusters(bool_dist,bool_eu_clust_df,cluster_colors,cluster_names,'bool_eu_cluster')

get_top_terms_per_cluster(ted_clean,count_vectorizer,count_eu_clust,count_eu_centroids,num_clusters,label='labels')
count_eu_clust_df=pd.DataFrame(count_eu_clust,columns=['cluster']).merge(pd.DataFrame.from_dict(cluster_df), on=['cluster'], how='left')
count_eu_clust_df=count_eu_clust_df.merge(pd.DataFrame(list(ted_clean['tags']),columns=['tags']),left_index=True, right_index=True)
count_eu_clust_df=count_eu_clust_df.merge(pd.DataFrame(list(ted_clean['labels'].apply(lambda x: 'o' if x=='popular' else 'x')),columns=['marker']),left_index=True, right_index=True)
plot_clusters(count_dist,count_eu_clust_df,cluster_colors,cluster_names,'count_eu_cluster')

#get_top_terms_per_cluster(ted_clean,tfidf_vectorizer,tfidf_eu_clust,tfidf_eu_centroids,num_clusters,label='labels')
#tfidf_eu_clust_df=pd.DataFrame(tfidf_eu_clust,columns=['cluster']).merge(pd.DataFrame.from_dict(cluster_df), on=['cluster'], how='left')
#tfidf_eu_clust_df=tfidf_eu_clust_df.merge(pd.DataFrame(list(ted_clean['tags']),columns=['tags']),left_index=True, right_index=True)
#plot_clusters(tfidf_dist,tfidf_eu_clust_df,cluster_colors,cluster_names,'tfidf_eu_cluster')
#
#get_top_terms_per_cluster(ted_clean,bool_vectorizer,bool_cos_clust,bool_cos_centroids,num_clusters,label='labels')
#bool_cos_clust_df=pd.DataFrame(bool_cos_clust,columns=['cluster']).merge(pd.DataFrame.from_dict(cluster_df), on=['cluster'], how='left')
#bool_cos_clust_df=bool_cos_clust_df.merge(pd.DataFrame(list(ted_clean['tags']),columns=['tags']),left_index=True, right_index=True)
#plot_clusters(bool_cosdist,bool_cos_clust_df,cluster_colors,cluster_names,'bool_cos_cluster')
#
#get_top_terms_per_cluster(ted_clean,count_vectorizer,count_cos_clust,count_cos_centroids,num_clusters,label='labels')
#count_cos_clust_df=pd.DataFrame(count_cos_clust,columns=['cluster']).merge(pd.DataFrame.from_dict(cluster_df), on=['cluster'], how='left')
#count_cos_clust_df=count_cos_clust_df.merge(pd.DataFrame(list(ted_clean['tags']),columns=['tags']),left_index=True, right_index=True)
#plot_clusters(count_cosdist,count_cos_clust_df,cluster_colors,cluster_names,'count_cos_cluster')
#
#get_top_terms_per_cluster(ted_clean,tfidf_vectorizer,tfidf_cos_clust,tfidf_cos_centroids,num_clusters,label='labels')
#tfidf_cos_clust_df=pd.DataFrame(tfidf_cos_clust,columns=['cluster']).merge(pd.DataFrame.from_dict(cluster_df), on=['cluster'], how='left')
#tfidf_cos_clust_df=tfidf_cos_clust_df.merge(pd.DataFrame(list(ted_clean['tags']),columns=['tags']),left_index=True, right_index=True)
#plot_clusters(tfidf_cosdist,tfidf_cos_clust_df,cluster_colors,cluster_names,'tfidf_cos_cluster')



#import the Vectorized data in CSV files as dataframe

bool_df.to_csv(path + 'bool_ted.csv', encoding='utf-8', index=False)
count_df.to_csv(path + 'count_ted.csv', encoding='utf-8', index=False)
tfidf_df.to_csv(path + 'tfidf_ted.csv', encoding='utf-8', index=False)


vect_count_ted = count_df
vect_bool_ted = bool_df
vect_tfidf_ted = tfidf_df

#Print First 5 rows
print(vect_count_ted.head())
print(vect_bool_ted.head())
print(vect_tfidf_ted.head())


#print shapes

print(vect_count_ted.shape)
print(vect_bool_ted.shape)
print(vect_tfidf_ted.shape)

#vect_count_ted.describe
#vect_bool_ted.describe
#vect_tfidf_ted.describe

#vect_count_ted.dtypes
#vect_bool_ted.dtypes
#vect_tfidf_ted.dtypes


print("Since the data is well-wrangled, lets get straight into creating models.\n")

###########################################################################
######
############## Model for determining popularity using Multinomial Naive Bayes using cross fold validation (10)
######
###########################################################################


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


#Note that if we are using CV there is no need to split data into test and train,
#all we need to do is seprate the labels only and have the train data as whole
#Then we user predict() to generate the labels again and then use accuracy_score,
#and confusion_matrix to check the accuracy of predicted data.

print("One of the requirements for creating a MNB model with crossfold validation is seprating labels So let's do that.\n")

## Save labels in a seprate series
vect_count_labels = vect_count_ted["labels"]
vect_bool_labels = vect_bool_ted["labels"]
vect_tfidf_labels = vect_tfidf_ted["labels"]

print("Other requirements for creating a MNB model with crossfold validation is having labels represented as array.So let's do that as well.\n")
#Convert the labels to array as that's what MNB methiond with CV requires
vect_count_labels_as_array= vect_count_labels.to_numpy()
vect_bool_labels_as_array= vect_bool_labels.to_numpy()
vect_tfidf_labels_as_array= vect_tfidf_labels.to_numpy()



#create new DF without any labels
vect_count_ted_no_labels = vect_count_ted.drop(["labels"], axis=1)
vect_bool_ted_no_labels = vect_bool_ted.drop(["labels"], axis=1)
vect_tfidf_ted_no_labels = vect_tfidf_ted.drop(["labels"], axis=1)




#verify that the labels columns are dropped
print(vect_count_ted_no_labels.head())
print(vect_bool_ted_no_labels.head())
print(vect_tfidf_ted_no_labels.head())


###########################################################################
######
############## Model for determining popularity using Multinomial Naive Bayes 
############## using cross fold validation (10) - COUNT vectorized data
######
###########################################################################

clf_count = MultinomialNB(alpha=1.0).fit(vect_count_ted_no_labels, vect_count_labels_as_array)

scores_count = cross_val_score(clf_count, vect_count_ted_no_labels, vect_count_labels_as_array, cv=10)

scores_count 
                                       

print("The mean score and the 95 percent confidence interval of the score estimate are hence given by: Accuracy: %0.2f (+/- %0.2f)" % (scores_count .mean(), scores_count .std() * 2))

Predicted_MNB_data_with_CV_Count = clf_count.predict(vect_count_ted_no_labels)
#optional lines to print predicted and actuals
#print("The predictions made for whether or not a ted talk is popular using MNB with CV=10 is:")
#print(Predicted_MNB_data_with_CV_Count)
#print("The actual labels are:")
#print(TestLabels)

#Note that once even compare and see where the mispredictions were made by comparing the two lists
#err_cnt = 0
#for i in range(0, len(y_test)):
#    if(y_test[i]==4 and y_pred[i]==1): compare the real with predicted
#        print(X_test[i])
#        err_cnt = err_cnt+1
#print("\n\n\berrors:", err_cnt)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We have TWO labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
cnf_matrix_count = confusion_matrix(vect_count_labels_as_array, Predicted_MNB_data_with_CV_Count)
print("\n\nConfusion matrix when using Multinomial Naive Bayes for predicting popularity of TED talks using Crossfold Validation method using count vectorized data:\n")
print(cnf_matrix_count)
#research how to make sense of confusion matrix when there are three labels

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#research  what predict_proba does
#print("__________________")
#print("__________________")
#print(np.round(MyModelNB.predict_proba(TestDF),2))

#Find accuracy of the model in test data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print accuracy
accuracy_of_model_predicted_data_CV_count= accuracy_score(vect_count_labels_as_array, Predicted_MNB_data_with_CV_Count)
print("__________________")

print("\nAccuracy when using Multinomial Naive Bayes for predicting popularity of TED talks using Crossfold validation method using count vectorized data: ", accuracy_of_model_predicted_data_CV_count,"\n\n")
print("__________________")
# print precision, recall, F1-score per each class/tag
print(classification_report(vect_count_labels_as_array, Predicted_MNB_data_with_CV_Count))



###########################################################################
######
############## Model for determining popularity using Multinomial Naive Bayes 
############## using cross fold validation (10) - Tfidf vectorized data
###########################################################################

clf_tfidf = MultinomialNB(alpha=1.0).fit(vect_tfidf_ted_no_labels, vect_tfidf_labels_as_array)

scores_tfidf = cross_val_score(clf_tfidf, vect_tfidf_ted_no_labels, vect_tfidf_labels_as_array, cv=10)

scores_tfidf 
                                      

print("The mean score and the 95 percent confidence interval of the score estimate are hence given by: Accuracy: %0.2f (+/- %0.2f)" % (scores_count.mean(), scores_count.std() * 2))

Predicted_MNB_data_with_CV_tfidf = clf_tfidf.predict(vect_tfidf_ted_no_labels)
#optional lines to print predicted and actuals
#print("The predictions made for whether or not a ted talk is popular using MNB with CV=10 is:")
#print(Predicted_MNB_data_with_CV_tfidf)
#print("The actual labels are:")
#print(TestLabels)

#Note that once even compare and see where the mispredictions were made by comparing the two lists
#err_cnt = 0
#for i in range(0, len(y_test)):
#    if(y_test[i]==4 and y_pred[i]==1): compare the real with predicted
#        print(X_test[i])
#        err_cnt = err_cnt+1
#print("\n\n\berrors:", err_cnt)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We have TWO labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
cnf_matrix_tfidf= confusion_matrix(vect_tfidf_labels_as_array, Predicted_MNB_data_with_CV_tfidf)
print("\n\nConfusion matrix when using Multinomial Naive Bayes for predicting popularity of TED talks using Crossfold Validation method using TFIDF vectorized data:\n")
print(cnf_matrix_tfidf)
#research how to make sense of confusion matrix when there are three labels

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#research  what predict_proba does
#print("__________________")
#print("__________________")
#print(np.round(MyModelNB.predict_proba(TestDF),2))

#Find accuracy of the model in test data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print accuracy
accuracy_of_model_predicted_data_CV_tfidf= accuracy_score(vect_tfidf_labels_as_array, Predicted_MNB_data_with_CV_tfidf)
print("__________________")
print("\nAccuracy when using Multinomial Naive Bayes for predicting popularity of TED talks using Crossfold Validation method using TFIDF vectorized data: ", accuracy_of_model_predicted_data_CV_tfidf,"\n\n")
print("__________________")
# print precision, recall, F1-score per each class/tag
print(classification_report(vect_tfidf_labels_as_array, Predicted_MNB_data_with_CV_tfidf))


###########################################################################
######
############## Model for determining popularity using Bernoullis Naive Bayes 
############## using cross fold validation (10) - Boolean count (binary) vectorized data
######
###########################################################################

clf_bool = BernoulliNB(alpha=1.0, binarize=1.0, class_prior=None, fit_prior=True).fit(vect_bool_ted_no_labels, vect_bool_labels_as_array)

scores_bool = cross_val_score(clf_bool, vect_bool_ted_no_labels, vect_bool_labels_as_array, cv=10)

scores_bool
                                       

print("The mean score and the 95 percent confidence interval of the score estimate are hence given by: Accuracy: %0.2f (+/- %0.2f)" % (scores_bool.mean(), scores_bool.std() * 2))

Predicted_BernNB_data_with_CV_bool = clf_bool.predict(vect_bool_ted_no_labels)
#optional lines to print predicted and actuals
#print("The predictions made for whether or not a ted talk is popular using BNB with CV=10 is:")
#print(Predicted_MNB_data_with_CV_Bool)
#print("The actual labels are:")
#print(vect_bool_labels_as_array)

#Note that once even compare and see where the mispredictions were made by comparing the two lists
#err_cnt = 0
#for i in range(0, len(y_test)):
#    if(y_test[i]==4 and y_pred[i]==1): compare the real with predicted
#        print(X_test[i])
#        err_cnt = err_cnt+1
#print("\n\n\berrors:", err_cnt)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We have TWO labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
cnf_matrix_bool = confusion_matrix(vect_bool_labels_as_array, Predicted_BernNB_data_with_CV_bool)
print("\n\nConfusion matrix when using Bernoulli's Naive Bayes for predicting popularity of TED talks using Crossfold validation method using boolean count (binary) vectorized data:\n")
print(cnf_matrix_bool)
#research how to make sense of confusion matrix when there are three labels

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#research  what predict_proba does
#print("__________________")
#print("__________________")
#print(np.round(MyModelNB.predict_proba(TestDF),2))

#Find accuracy of the model in test data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print accuracy
accuracy_of_model_predicted_data_CV_bool= accuracy_score(vect_bool_labels_as_array, Predicted_BernNB_data_with_CV_bool)
print("__________________")

print("\nAccuracy when using Bernoulli's Naive Bayes for predicting popularity of TED talks using Crossfold validation method using boolean count (binary) vectorized data: ", accuracy_of_model_predicted_data_CV_bool,"\n\n")
print("__________________")
# print precision, recall, F1-score per each class/tag
print(classification_report(vect_bool_labels_as_array, Predicted_BernNB_data_with_CV_bool))


###########################################################################
######
############## Model for determining popularity using SVM (Linear SVC)
############## using cross fold validation (10) - Count vectorized data
######
###########################################################################

clf_count_svm = LinearSVC(C=10).fit(vect_count_ted_no_labels, vect_count_labels_as_array)

scores_count_svm= cross_val_score(clf_count_svm, vect_count_ted_no_labels, vect_count_labels_as_array, cv=10)

scores_count_svm
                                       

print("The mean score and the 95 percent confidence interval of the score estimate are hence given by: Accuracy: %0.2f (+/- %0.2f)" % (scores_count_svm.mean(), scores_count_svm.std() * 2))

Predicted_SVM_data_with_CV_count_svm = clf_count_svm.predict(vect_count_ted_no_labels)
#optional lines to print predicted and actuals
#print("The predictions made for whether or not a ted talk is popular using BNB with CV=10 is:")
#print(Predicted_MNB_data_with_CV_Bool)
#print("The actual labels are:")
#print(vect_bool_labels_as_array)

#Note that once even compare and see where the mispredictions were made by comparing the two lists
#err_cnt = 0
#for i in range(0, len(y_test)):
#    if(y_test[i]==4 and y_pred[i]==1): compare the real with predicted
#        print(X_test[i])
#        err_cnt = err_cnt+1
#print("\n\n\berrors:", err_cnt)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We have TWO labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
cnf_matrix_count_svm  = confusion_matrix(vect_count_labels_as_array, Predicted_SVM_data_with_CV_count_svm)
print("\n\nConfusion matrix when using SVM for predicting popularity of TED talks using Crossfold validation method using count vectorized data:\n")
print(cnf_matrix_count_svm)
#research how to make sense of confusion matrix when there are three labels

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#research  what predict_proba does
#print("__________________")
#print("__________________")
#print(np.round(MyModelNB.predict_proba(TestDF),2))

#Find accuracy of the model in test data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print accuracy
accuracy_of_model_predicted_data_CV_count_svm= accuracy_score(vect_count_labels_as_array, Predicted_SVM_data_with_CV_count_svm)
print("__________________")
print("\n Accuracy when using SVM for predicting popularity of TED talks using Crossfold validation method using count vectorized data: ", accuracy_of_model_predicted_data_CV_count_svm,"\n\n")
print("__________________")
# print precision, recall, F1-score per each class/tag
print(classification_report(vect_count_labels_as_array, Predicted_SVM_data_with_CV_count_svm))


###########################################################################
######
############## Model for determining popularity using Linear SVM
############## using cross fold validation (10) - TFIDF vectorized data
######
###########################################################################

clf_tfidf_svm = LinearSVC(C=10).fit(vect_tfidf_ted_no_labels, vect_tfidf_labels_as_array)

scores_tfidf_svm= cross_val_score(clf_tfidf_svm, vect_tfidf_ted_no_labels, vect_tfidf_labels_as_array, cv=10)

scores_tfidf_svm
                                       

print("The mean score and the 95 percent confidence interval of the score estimate are hence given by: Accuracy: %0.2f (+/- %0.2f)" % (scores_tfidf_svm.mean(), scores_tfidf_svm.std() * 2))

Predicted_SVM_data_with_CV_tfidf_svm = clf_tfidf_svm.predict(vect_tfidf_ted_no_labels)
#optional lines to print predicted and actuals
#print("The predictions made for whether or not a ted talk is popular using SVM with CV=10 is:")
#print(Predicted_MNB_data_with_CV_tfidf_svm)
#print("The actual labels are:")
#print(vect_tfidf_labels_as_array)

#Note that once even compare and see where the mispredictions were made by comparing the two lists
#err_cnt = 0
#for i in range(0, len(y_test)):
#    if(y_test[i]==4 and y_pred[i]==1): compare the real with predicted
#        print(X_test[i])
#        err_cnt = err_cnt+1
#print("\n\n\berrors:", err_cnt)

## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We have TWO labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
cnf_matrix_tfidf_svm  = confusion_matrix(vect_tfidf_labels_as_array, Predicted_SVM_data_with_CV_tfidf_svm)
print("\n\nConfusion matrix when using SVM for predicting popularity of TED talks using Crossfold validation method using TFIDF vectorized data:\n")
print(cnf_matrix_tfidf_svm)
#research how to make sense of confusion matrix when there are three labels

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
#research  what predict_proba does
#print("__________________")
#print("__________________")
#print(np.round(MyModelNB.predict_proba(TestDF),2))

#Find accuracy of the model in test data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print accuracy
accuracy_of_model_predicted_data_CV_tfidf_svm = accuracy_score(vect_tfidf_labels_as_array, Predicted_SVM_data_with_CV_tfidf_svm)
print("__________________")
print("\n Accuracy when using SVM for predicting popularity of TED talks using Crossfold validation method using TFIDF vectorized data: ", accuracy_of_model_predicted_data_CV_tfidf_svm,"\n\n")
print("__________________")
# print precision, recall, F1-score per each class/tag
print(classification_report(vect_tfidf_labels_as_array, Predicted_SVM_data_with_CV_tfidf_svm))

ted_main_df= ted_clean

vectorizer = TfidfVectorizer(stop_words="english",
                        use_idf=True,
                        ngram_range=(1,1), # considering only 1-grams
                        min_df = 0.05,     # cut words present in less than 5% of documents
                        max_df = 0.3)      # cut words present in more than 30% of documents 
t0 = time()

tfidf = vectorizer.fit_transform(ted_main_df['transcript'])
print("done in %0.3fs." % (time() - t0))

def rank_words(terms, feature_matrix):
    sums = feature_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(terms):
        data.append( (term, sums[0,col]) )
    ranked = pd.DataFrame(data, columns=['term','rank']).sort_values('rank', ascending=False)
    return ranked

ranked = rank_words(terms=vectorizer.get_feature_names(), feature_matrix=tfidf)

fig, ax = plt.subplots(figsize=(6,10), ncols=1, nrows=1)
sns.barplot(x='rank',y='term',data=ranked[:20], palette='Reds_r', ax=ax)

dic = {ranked.loc[i,'term'].upper(): ranked.loc[i,'rank'] for i in range(0,len(ranked))}

#from wordcloud import WordCloud
#wordcloud = WordCloud(background_color='white',
                      #max_words=100,
                      #colormap='Reds').generate_from_frequencies(dic)
#fig = plt.figure(1,figsize=(12,15))
#plt.imshow(wordcloud,interpolation="bilinear")
#plt.axis('off')
#plt.show()
                      

from sklearn.decomposition import LatentDirichletAllocation

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics,random_state=0)

topics = lda.fit_transform(tfidf)
top_n_words = 5
t_words, word_strengths = {}, {}
for t_id, t in enumerate(lda.components_):
    t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
    word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
t_words      

fig, ax = plt.subplots(figsize=(7,15), ncols=2, nrows=5)
plt.subplots_adjust(
    wspace  =  0.5,
    hspace  =  0.5
)
c=0
for row in range(0,5):
    for col in range(0,2):
        sns.barplot(x=word_strengths[c], y=t_words[c], color="red", ax=ax[row][col])
        c+=1
plt.show()     

from sklearn.decomposition import NMF

n_topics = 10
nmf = NMF(n_components=n_topics,random_state=0)

topics = nmf.fit_transform(tfidf)
top_n_words = 5
t_words, word_strengths = {}, {}
for t_id, t in enumerate(nmf.components_):
    t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
    word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
t_words    


fig, ax = plt.subplots(figsize=(7,15), ncols=2, nrows=5)
plt.subplots_adjust(
    wspace  =  0.5,
    hspace  =  0.5
)
c=0
for row in range(0,5):
    for col in range(0,2):
        sns.barplot(x=word_strengths[c], y=t_words[c], color="red", ax=ax[row][col])
        c+=1
plt.show()       


from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('tfidf', vectorizer),
    ('nmf', nmf)
])

document_id = 4
t = pipe.transform([ted_main_df['transcript'].iloc[document_id]]) 
print('Topic distribution for document #{}: \n'.format(document_id),t)
print('Relevant topics for document #{}: \n'.format(document_id),np.where(t>0.01)[1])
print('\nTranscript:\n',ted_main_df['transcript'].iloc[document_id][:500],'...')

talk = ted_main_df[ted_main_df['url']==ted_main_df['url'].iloc[document_id]]
print('\nTrue tags from ted_main.csv: \n',talk['tags'])

t = pipe.transform(ted_main_df['transcript']) 
t = pd.DataFrame(t, columns=[str(t_words[i]) for i in range(0,10)])
t.head()

new_t = t.melt()

# fig = plt.figure(1,figsize=(12,6))
fig, ax = plt.subplots(figsize=(12,6), ncols=1, nrows=1)
sns.violinplot(x="variable", y="value", data=new_t, palette='Reds', ax=ax)
# plt.xticks(rotation=75)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor") 

plt.show()