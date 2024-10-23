import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

# Specify the path to your JSON file
json_file_path = 'C:\\Users\\okwuchi.uzoigwe\\Downloads\\review_queries_ganesh_gan_15d563baa3283ba1_1626113046000.json'

# Read movie reviews from the JSON file
def read_reviews_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reviews_data = json.load(file)
        reviews = [(review['content'], review['rating']) for review in reviews_data]
    return reviews

# Download the 'punkt' tokenizer if not already downloaded
nltk.download('punkt')

# Function to extract features from the given text
def extract_features(text):
    words = word_tokenize(text)
    return dict([(word, True) for word in words])

# Function to assign sentiment labels based on ratings
def assign_sentiment_label(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# Function to train a Naive Bayes classifier on the reviews dataset
def train_naive_bayes_classifier(reviews):
    labeled_reviews = [(extract_features(text), assign_sentiment_label(rating)) for text, rating in reviews]
    return NaiveBayesClassifier.train(labeled_reviews)

# Function to perform sentiment analysis using Naive Bayes
def sentiment_analyze_naive_bayes(sentiment_text, classifier):
    features = extract_features(sentiment_text)
    sentiment = classifier.classify(features)
    return sentiment

# Example usage:
# Read reviews from the JSON file
reviews_from_json = read_reviews_from_json(json_file_path)

# Example usage:
text = "This is product is so good, I love it so much"
rating = 5.0  # Assume you have a specific rating for the example review

# Assign sentiment label based on rating
sentiment_label = assign_sentiment_label(rating)
print("Rating:", rating)
print("Assigned Sentiment Label:", sentiment_label)

# Train Naive Bayes classifier
nb_classifier = train_naive_bayes_classifier(reviews_from_json)

# Perform sentiment analysis using Naive Bayes
naive_bayes_sentiment = sentiment_analyze_naive_bayes(text, nb_classifier)
print("Naive Bayes Sentiment:", naive_bayes_sentiment)
