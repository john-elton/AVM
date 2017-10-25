#import math
#import pandas as pd
#import numpy as np
#import scipy as sp
#import urllib2
#import base64
#from sklearn.metrics import confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.preprocessing import scale
#from sklearn import metrics
#from copy import deepcopy
#import mord
#import pickle
#
#
#user = 'john'
#password = "Qb7ousmjTHFmB8e3og0jloDHX4AQiigG"
## Hard-coded based on 5 different webpages
#for i in range(1,6):
##    url = "http://avmrails-test.herokuapp.com/api/v1/valuations?page=" + str(i)
#    url = "https://avmrails-test.herokuapp.com/api/v1/user_valuations?page=" + str(i)
#    request = urllib2.Request(url)
#    base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
#    request.add_header("Authorization", "Basic %s" % base64string)   
#    result = urllib2.urlopen(request)
#    if i==1:
#        df = pd.read_json(result)
#    else:
#        df_new = pd.read_json(result)
#        df = pd.concat([df,df_new])
#df = df.reset_index(drop=True)
#
#ndim = len(df['valuations'][0])
#objs = [df, pd.DataFrame(df['valuations'].tolist()).iloc[:, :ndim]]
#df_orig = pd.concat(objs, axis=1).drop('valuations', axis=1)
##df_orig = df_orig.drop('media',1)
##df_orig = df_orig.fillna('NA')
##df_orig.to_csv('orig.csv', sep=',',encoding='utf-8')
#
##df_orig['bathrooms'] = df_orig['full_bathrooms'] + df_orig['half_bathrooms']
##
##df_orig['year_built'] = df_orig['year_built'].astype('int')
#
#N_orig = df_orig.shape[0]
#
#
#df_orig.to_csv('user_val.csv', sep=',',encoding='utf-8')




import math
import pandas as pd
import numpy as np
import scipy as sp
import base64
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import scale
from sklearn import metrics
from copy import deepcopy
import mord
import pickle
import jenkspy


PYTHON2 = 0

if PYTHON2 == 1:
    import urllib2
else:
    from urllib.request import urlopen, Request


#from bs4 import BeautifulSoup
#import re
#from nltk.corpus import stopwords
#import nltk.data
##nltk.download()     # uncomment when running for first time
#tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle') 
#
#import gensim
#from gensim.models import Word2Vec
#from gensim.models import word2vec
#from gensim.models import Doc2Vec
#from gensim.models import doc2vec
#from gensim.models import Phrases
#import logging


user = 'john'
password = "Qb7ousmjTHFmB8e3og0jloDHX4AQiigG"
# Hard-coded based on 5 different webpages
for i in range(1,2):
#    url = "http://avmrails-test.herokuapp.com/api/v1/listings?page=" + str(i)
    #url = "https://avmrails-test.herokuapp.com/api/v1/user_valuations?page=" + str(i)
    
    #url = "https://avmrails-test.herokuapp.com/api/v1/valuations/29068"
    #url = "https://avmrails-test.herokuapp.com/api/v1/valuations/24048"
    url = "https://avmrails-test.herokuapp.com/api/v1/valuations/28884"

    
    if PYTHON2 == 1:
        request = urllib2.Request(url)
    
        base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
    
    
        request.add_header("Authorization", "Basic %s" % base64string)   
        result = urllib2.urlopen(request)

    else:  
        request = Request(url)    

        credentials = ('%s:%s' % (user, password))
        encoded_credentials = base64.b64encode(credentials.encode('ascii'))
        request.add_header('Authorization', 'Basic %s' % encoded_credentials.decode("ascii"))
        
        with urlopen(request) as response:
            result = response.read()
            
    
    if i==1:
        #df = pd.read_json(result)
        df = pd.read_json(result,orient='index')
    else:
        df_new = pd.read_json(result)
        df = pd.concat([df,df_new])
df = df.reset_index(drop=True)

# might have to change between 'valuation' and 'valutions'

df['listing'][0]['id_listing'] = df['listing'][0].pop('id')

ndim = len(df['listing'][0])
objs = [df, pd.DataFrame(df['listing'].tolist()).iloc[:, :ndim]]
df_listing = pd.concat(objs, axis=1).drop('listing', axis=1)


#modify listing and comp dataframes so they can be combined
df_listing['bathrooms'] = df_listing['baths_full'] + df_listing['baths_half']

#df['listing'][0]['id_listing'] = df['listing'][0].pop('id')


df_comps = df['comparables'].to_dict()
df_comps = df_comps[0]
df_comps = pd.DataFrame(df_comps, columns = df_comps[0].keys())

#df_orig.to_csv('user_val.csv', sep=',',encoding='utf-8')

df_comps['bathrooms'] = df_comps['full_bathrooms'] + df_comps['half_bathrooms']
df_comps['year_built'] = df_comps['year_built'].astype('int')
df_comps['id_listing'] = df_comps['id']

frames = [df_listing[['id_listing','asking_price','bedrooms','bathrooms','subdivision','city','year_built','picture_count','public_remarks','closing_price','adjusted_sqft']],df_comps[['id_listing','asking_price','bedrooms','bathrooms','subdivision','city','year_built','picture_count','public_remarks','closing_price','adjusted_sqft']]]

df_combined = pd.concat(frames).reset_index(drop=True)


df_combined = df_combined.fillna('NA')


# Impute missing numerical values with the mean value
# May have to tailor this based on various things going wrong
df_combined.loc[df_combined['closing_price'] == 'NA', 'closing_price'] = np.mean(df_combined[df_combined['closing_price'] != 'NA']['closing_price'])
df_combined.loc[df_combined['adjusted_sqft'] == 0, 'adjusted_sqft'] = np.mean(df_combined[df_combined['adjusted_sqft'] != 0]['adjusted_sqft'])



# If doing NLP
df_combined = df_combined[df_combined['public_remarks'] != 'NA']
# Else
#df_combined = df_combined.drop('public_remarks',1)


# If only 2 labels desired
#df_combined.loc[df_combined['condition_rating'] == 2, 'condition_rating'] = 1
#df_combined.loc[df_combined['condition_rating'] == 4, 'condition_rating'] = 3
#df_combined.loc[df_combined['condition_rating'] == 3, 'condition_rating'] = 2


N = df_combined.shape[0]

# load the models from disk
filename1 = 'RF_model.sav'
RF_model = pickle.load(open(filename1, 'rb'))

filename2 = 'classifier_comb2_model.sav'
classifier_comb2_model = pickle.load(open(filename2, 'rb'))

filename3 = 'count_vectorizer.sav'
count_vectorizer = pickle.load(open(filename3, 'rb'))

filename4 = 'tfidf_transformer.sav'
tfidf_transformer = pickle.load(open(filename4, 'rb'))


#df_combined = df_combined.drop('condition_rating',1) 

text_column = 'public_remarks'
# one-hot encode categorical features with dummy columns, leaving one out for each
#cols_to_transform = ['subdivision','city']
cols_to_transform = []

df_dummies = pd.get_dummies(data=df_combined,columns = cols_to_transform,drop_first=True )
df_dummies = df_dummies.drop(['subdivision','city'],1)

x_test = df_dummies.reset_index(drop=True)

listing_identifiers = x_test[['id_listing','asking_price','adjusted_sqft','year_built']]
x_test = x_test.drop(['id_listing','asking_price'],1)

    
if PYTHON2 == 1:
    x_test_1 = np.char.array(x_test[text_column].values)
else:
    x_test_1 = np.array(x_test[text_column].values)


x_test = x_test.drop('public_remarks',1)


#count_vectorizer = CountVectorizer(stop_words = 'english', ngram_range=(1, 2), max_features = None)
#tfidf_transformer = TfidfTransformer()

# Get counts and td-idf for the test sets
test_counts = count_vectorizer.transform(x_test_1)
x_test_tfidf = tfidf_transformer.transform(test_counts)

xtt_np = x_test.values.astype(np.float)

x_test_comb = np.hstack([xtt_np, x_test_tfidf.toarray()])

RF_predictions = RF_model.predict(x_test_comb)

classifier_comb2_predictions = classifier_comb2_model.predict(x_test_comb)

breaks = jenkspy.jenks_breaks(RF_predictions, nb_class = 4)
RF_pred_labels = np.zeros((N,),dtype=int)
for k in range(N):
    if RF_predictions[k] < breaks[1]:
        RF_pred_labels[k] = 1
    elif RF_predictions[k] < breaks[2]:
        RF_pred_labels[k] = 2
    elif RF_predictions[k] < breaks[3]:
        RF_pred_labels[k] = 3
    else:
        RF_pred_labels[k] = 4
        
listing_identifiers['clss_comb2_preds'] = classifier_comb2_predictions
listing_identifiers['RF_pred_labels'] = RF_pred_labels
listing_identifiers['RF_predictions'] = RF_predictions

AVM_output = listing_identifiers.drop(['asking_price','adjusted_sqft','year_built','RF_pred_labels','RF_predictions'],1)

listing_identifiers.to_csv('listing_identifiers2.csv', sep=',',encoding='utf-8')

zzz = 1







