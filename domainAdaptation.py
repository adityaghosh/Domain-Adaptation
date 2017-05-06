"""
@author: Aditya Ghosh
Description:
Here we use domain adaptation as a means to predict from the categories "Politics", "Sports" and "Technology" 
for documents in the BBC dataset using the 20Newsgroup dataset as a source.
"""

import sys
import numpy as np
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
import copy
from sklearn.naive_bayes import MultinomialNB

########################-----------------------DATA LOAD-----------------------#########################
# Load classifier based on system argument
# [MultinomialNB, LogisticRegression]
classifier = MultinomialNB()
try:
	print("Using Classifier: "+sys.argv[1])
except:
	print("No system arguments! Please give the classifier you want to use [MultinomialNB, LogisticRegression]. Now choosing MultinomialNB by default.")
else:	
	if (sys.argv[1] == "LogisticRegression"):
		classifier = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
	else:
		classifier = MultinomialNB()

# Here we only 3 categories
categories = [ 'politics','sport','tech' ]

print("Loading 20 newsgroups dataset as Source and BBC News as Target with categories:")
print(categories)
try:
	# Train on 20_NewsGroup Data
	data_source = load_files("20_newsgroups", categories=categories ,random_state=42, encoding='latin1')
	# BBC as test data
	data_target = load_files("bbc_full", categories=categories ,random_state=42, encoding='latin1')
	print('data loaded')
except:
	print("Data folder does not exist. Or not cleaned.")
	exit()

# order of labels in `target_names` can be different from `categories`
target_names = data_source.target_names

def size_mb(docs):
    return sum(len(s.encode('latin1')) for s in docs) / 1e6

data_source_size_mb = size_mb(data_source.data)
data_target_size_mb = size_mb(data_target.data)

print("%d documents - %0.3fMB (Source dataset - Training)" % (
    len(data_source.data), data_source_size_mb))
print("%d documents - %0.3fMB (Target dataset - Prediction)" % (
    len(data_target.data), data_target_size_mb))
print("%d categories" % len(categories))
########################-----------------------DATA LOAD-----------------------#########################

########################-------------------FEATURE EXTRACTION-------------------########################
print("Extracting features from the Source data")
vectorizer_source =  CountVectorizer(ngram_range=(1, 2),stop_words='english')
X_source_train = vectorizer_source.fit_transform(data_source.data)    
print("n_samples: %d, n_features: %d" % X_source_train.shape)

print("Extracting features from the Target data")
vectorizer_target = CountVectorizer(ngram_range=(1, 2),stop_words='english')
Y_target_test =  vectorizer_target.fit_transform(data_target.data)
print("n_samples: %d, n_features: %d" % Y_target_test.shape)

feature_names_source = vectorizer_source.get_feature_names()
print("Features in Source: %d" % len(feature_names_source))
print(np.asarray(feature_names_source))

feature_names_target = vectorizer_target.get_feature_names()
print("Features in Target: %d" % len(feature_names_target))
print(np.asarray(feature_names_target))

# Find common features
a = set(feature_names_target)
b = set(feature_names_source)
feature_names_common = list(a.intersection(b))
print("Common Features: %d" % len(feature_names_common))
print(np.asarray(feature_names_common))
########################-------------------FEATURE EXTRACTION-------------------########################


################-----------------BUILD CLASSIFIER USING SOURCE FEATURES-----------------################
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),
                                              vocabulary=feature_names_source,
                                              stop_words='english')),
                      ('clf',  classifier)
])
text_clf = text_clf.fit(data_source.data, data_source.target)
predicted = text_clf.predict(data_target.data)
acc = np.mean(predicted == data_target.target)
print("Accuracy before domain adaptation: %f" % (acc*100))

print(metrics.classification_report(data_target.target, predicted,
     target_names=data_target.target_names))
################-----------------BUILD CLASSIFIER USING SOURCE FEATURES-----------------################


################-----------------BUILD CLASSIFIER USING COMMON FEATURES-----------------################
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),
                                              vocabulary=feature_names_common,
                                              stop_words='english')),
                      ('clf',  	classifier)
])
text_clf = text_clf.fit(data_source.data, data_source.target)
predicted = text_clf.predict(data_target.data)
acc = np.mean(predicted == data_target.target)
print("Accuracy using only common feature set: %f" % (acc*100))

print(metrics.classification_report(data_target.target, predicted,
     target_names=data_target.target_names))
################-----------------BUILD CLASSIFIER USING COMMON FEATURES-----------------################

############----------CONFIDENT PREDICTIONS MADE MODEL TRAINED USING COMMON FEATURES---------###########
# Find the 130 = 10% of the most confident predictions in the target domain
confident_target_data=[]
confident_target_target = []
max_probs = []
for sample in data_target.data:
    prediction_prob = text_clf.predict_proba([sample])[0]
    max_prob = max(prediction_prob)
    category, = np.where( prediction_prob == max_prob)
    if (len(max_probs) <=130):
        confident_target_data.append(sample)
        confident_target_target.append(category[0])
	max_probs.append(max_prob)	
    else:
	if(min(max_probs) < max_prob):
		index = max_probs.index(min(max_probs))
		del confident_target_data[index]
		del confident_target_target[index]
		del max_probs[index] 		
		confident_target_data.append(sample)
		confident_target_target.append(category[0])
		max_probs.append(max_prob)

#prediction_prob = text_clf.predict_proba(data_target.data)
print("Number of documents with confident predictions - %d" % len(confident_target_data))
############----------CONFIDENT PREDICTIONS MADE MODEL TRAINED USING COMMON FEATURES---------###########

################---------------MOVE CONFIDENT PREDICTIONS TO SOURCE DOMAIN--------------################
# Add the confident features to the common feature set
vectorizer_confident_target = CountVectorizer(ngram_range=(1, 2),stop_words='english')
temp =  vectorizer_confident_target.fit_transform(confident_target_data)
confident_target_features = vectorizer_confident_target.get_feature_names()

set2 = set(feature_names_common)
set1 = set(confident_target_features)
set3 = set2.union(set1) 
print("Number of Common Features: %d" % len(set2))
print("Number of Confident Target features: %d" % len(set1))
print("Number of Unique Feautres after combining the above: %d" % len(set3))
new_vocab = list(set3)

# Move the confident predictions to the training set
c = copy.deepcopy(data_source.data)
c.extend(confident_target_data)

# Update the class of the confidently predicted documents as labeled
d = copy.deepcopy(data_source.target)
d = np.append(d,confident_target_target)
################---------------MOVE CONFIDENT PREDICTIONS TO SOURCE DOMAIN--------------################

#######-----BUILD CLASSIFIER USING COMMON+FEATURES FROM CONFIDENT PREDICTION IN TARGET DOMAIN----#######
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),
                                              vocabulary=new_vocab,
                                              stop_words='english')),
                      ('clf',  classifier)
])

text_clf = text_clf.fit(c, d)
predicted = text_clf.predict(data_target.data)
acc = np.mean(predicted == data_target.target)
print("Accuracy after domain adaptation: %f" % (acc*100))

print(metrics.classification_report(data_target.target, predicted,
     target_names=data_target.target_names))
#######-----BUILD CLASSIFIER USING COMMON+FEATURES FROM CONFIDENT PREDICTION IN TARGET DOMAIN----#######


