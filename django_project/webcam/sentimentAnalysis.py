testDataSet = []
c = 0
import csv
corpusFile1 = "/Users/apple/Desktop/convertcsv/convertcsv-2.csv"
corpusFile2 = "/Users/apple/Desktop/convertcsv/convertcsv-9.csv"
with open(corpusFile1,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            testDataSet.append({"text": row[3], "label":row[1]})

c=0
with open(corpusFile2,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            testDataSet.append({"text": row[3], "label":row[1]})

print(testDataSet)

corpusFile1 = "/Users/apple/Desktop/convertcsv/convertcsv-3.csv"
tweetDataFile = "/Users/apple/Desktop/convertcsv/tweetfile.csv"
corpusFile2 = "/Users/apple/Desktop/convertcsv/convertcsv-4.csv"
corpusFile3 = "/Users/apple/Desktop/convertcsv/convertcsv-5.csv"
corpusFile4 = "/Users/apple/Desktop/convertcsv/convertcsv-6.csv"
corpusFile5 = "/Users/apple/Desktop/convertcsv/convertcsv-7.csv"
corpusFile6 = "/Users/apple/Desktop/convertcsv/convertcsv-8.csv"
corpusFile7 = "/Users/apple/Desktop/convertcsv/convertcsv-1.csv"
import csv
import time

corpus = []
trainingDataSet = []
c = 0

with open(corpusFile1,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})

c=0
with open(corpusFile2,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})
c=0
with open(corpusFile3,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})
c=0
with open(corpusFile4,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})

c=0
with open(corpusFile5,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})

c=0
with open(corpusFile6,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})

c=0
with open(corpusFile7,'r') as csvfile:
    lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
    for row in lineReader:
        c = c + 1
        if(c>1):
            corpus.append({"tweet_id":row[0], "label":row[1], "text": row[3]})

for tweet in corpus:
    trainingDataSet.append(tweet)

with open(tweetDataFile,'w') as csvfile:
    linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
    for tweet in trainingDataSet:
        try:
            linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"]])
        except Exception as e:
            print(e)



import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets

    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = re.sub('[.]*','',tweet)
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]

tweetProcessor = PreProcessTweets()
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)
print(preprocessedTestSet)
preprocessedTrainingSet = tweetProcessor.processTweets(trainingDataSet)
print(preprocessedTrainingSet)

import nltk

def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
#     print(wordlist)
    word_features = wordlist.keys()
#     print(word_features)
    return word_features


def extract_features(tweet):
    tweet_words=set(tweet)
    print(tweet_words)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features



# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

from sklearn.metrics import accuracy_score
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def NBayesClassFinder(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs

def NaiveBayesEmotion(text):
    emotion = NBayesClassFinder(text)
    if(emotion['compound']>=0.5):
        sentiment = 'happy'
    elif(emotion['compound']>-0.5):
        sentiment = 'neutral'
    else:
        sentiment = 'sad'
    return sentiment

def findOutput():
    x = "I am very happy"
    testDataSet=[]
    testDataSet.append({"text": x, "label":""})
    tweetProcessor = PreProcessTweets()
    preprocessedTestSet = tweetProcessor.processTweets(testDataSet)
    NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
    print(NBResultLabels)
