import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

nltk.download('movie_reviews')
nltk.download('punkt')

# Function to extract features from the given text
def extract_features(text):
    words = word_tokenize(text)
    return dict([(word, True) for word in words])

# Function to train a Naive Bayes classifier on the movie reviews dataset
def train_naive_bayes_classifier():
    t = nltk.data.find('corpora/movie_reviews')
    print ("Movie review1:", t)
    print ("Movie review:", movie_reviews)
    positive_reviews = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in movie_reviews.fileids('pos')]
    negative_reviews = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in movie_reviews.fileids('neg')]
    train_set = positive_reviews + negative_reviews
    return NaiveBayesClassifier.train(train_set)

# Function to perform sentiment analysis using Naive Bayes
def sentiment_analyze_naive_bayes(sentiment_text, classifier):
    features = extract_features(sentiment_text)
    sentiment = classifier.classify(features)
    return sentiment

# Example usage:
text = "I hate this so much."
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores(text)

if score['compound'] >= 0.05:
    print("VADER Sentiment: Positive")
else:
    print("VADER Sentiment: Negative")

# Train Naive Bayes classifier
nb_classifier = train_naive_bayes_classifier()

# Perform sentiment analysis using Naive Bayes
naive_bayes_sentiment = sentiment_analyze_naive_bayes(text, nb_classifier)
print("Naive Bayes Sentiment:", naive_bayes_sentiment)
