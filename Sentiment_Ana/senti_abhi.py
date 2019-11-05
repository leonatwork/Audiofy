# Authenticating python script
import twitter
twitter_api = twitter.Api(consumer_key= 'random_one',
                        consumer_secret='random_two',
                        access_token_key='random_three',
                        access_token_secret='random_four')
print(twitter_api.VerifyCredentials())

# Function to build the test set
def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count = 100)
        
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None
    
search_term = input("Enter a search keyword:")
search_term = str(search_term)
print(search_term)
testDataSet = buildTestSet(search_term)

print(testDataSet[0:4])

# Function to build the training data set
def buildTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time
    
    corpus = []
    
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2], "label":row[1], "topic":row[0]})
            
    rate_limit = 180
    sleep_time = 900/180
    
    trainingDataSet = []
    
    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched" + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            time.sleep(sleep_time) 
        except: 
            continue
    # now we write them to the empty CSV file
    with open(tweetDataFile,'wb') as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet

corpusFile = "/home/abhijeet/corpus.csv"
tweetDataFile = "/home/abhijeet/tweetDataFile.csv"

trainingData = buildTrainingSet(corpusFile, tweetDataFile)
print(trainingData)

# Pre-processing the training and test data set to remove noise and unwanted entities
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
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]

tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
print(preprocessedTrainingSet)
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)
print(preprocessedTestSet)