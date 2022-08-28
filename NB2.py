import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import re #Regular expression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import accuracy_score

##This dataset contains information of users in a social network. This social network has several business clients which can post ads on it. One of the clients has a car company which has just launched a luxury SUV for a ridiculous price. Build a Bernoulli Naïve Bayes model using this dataset and classify which of the users of the social network are going to purchase this luxury SUV. 1 implies that there was a purchase and 0 implies there wasn’t a purchase.

car = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/NB_Car_Ad.csv")

car.head()
car = pd.get_dummies(car,columns = ['Gender'],drop_first = True)
x = car.drop(columns = ['Purchased'])
y = car.Purchased.values

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.25)

from sklearn.naive_bayes import GaussianNB
model_2 = GaussianNB().fit(xtrain,ytrain)
predict_y = model_2.predict(xtest)
predict_y

accuracyscore_2 = accuracy_score(ytest, predict_y) 
accuracyscore_2


##In this case study, you have been given Twitter data collected from an anonymous twitter handle. With the help of a Naïve Bayes model, predict if a given tweet about a real disaster is real or fake.

tweets = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/Disaster_tweets_NB.csv")

stop_words = []
# Load the custom built Stopwords
with open("/Users/jobinsamuel/Desktop/360Clses/Textminiing/Datasets NLP/stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

tweets.text = tweets.text.apply(cleaning_text)


# removing empty rows
tweets = tweets.loc[tweets.text != " ",:]


# splitting data into train and test data sets 

tweet_train, tweet_test = train_test_split(tweets, test_size = 0.25)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweets.text)


# Defining BOW for all tweets
all_tweet_matrix = tweet_bow.transform(tweets.text)

# For training tweets
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing tweets
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire tweets
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train tweets
traintw_tfidf = tfidf_transformer.transform(train_tweet_matrix)
traintw_tfidf.shape # (row, column)

# Preparing TFIDF for test tweets
testtw_tfidf = tfidf_transformer.transform(test_tweet_matrix)
testtw_tfidf.shape #  (row, column)


# Preparing a naive bayes model on training data set 


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(traintw_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(testtw_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(traintw_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

#Smoothening using laplace
#Model -2
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(traintw_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(testtw_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(traintw_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap



