import json
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import accuracy_score

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
    if isinstance(rating, (int, float)):
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'
    else:
        # Handle the case where rating is not a valid number
        return 'Unknown'

# Function to train a Naive Bayes classifier on the reviews dataset
def train_naive_bayes_classifier(reviews):
    labeled_reviews = [(extract_features(text), assign_sentiment_label(rating)) for text, rating in reviews]
    return NaiveBayesClassifier.train(labeled_reviews)

# Function to evaluate the accuracy of the classifier on a separate test set
def evaluate_classifier(classifier, test_set):
    
    predicted_labels = [classifier.classify(extract_features(text)) for text, _ in test_set]
    true_labels = [label for _, label in test_set]

    # Convert both predicted_labels and true_labels to strings
    predicted_labels = [str(label) for label in predicted_labels]

    true_labels = [str(label) for label in true_labels]

    # Combine extracted features and true labels
    labeled_test_set = [(extract_features(text), label) for text, label in test_set]

    print("test_set:", labeled_test_set)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print("accuracy:", accuracy)
    cm = ConfusionMatrix(true_labels, predicted_labels)
    print("cm:", cm)

    return accuracy, cm

# Example usage:
# Read reviews from the JSON file
reviews_from_json = read_reviews_from_json(json_file_path)

# Example usage:
# Split the dataset into a training set and a testing set
train_size = int(0.8 * len(reviews_from_json))
train_set = reviews_from_json[:train_size]
test_set = reviews_from_json[train_size:]

# Train Naive Bayes classifier
nb_classifier = train_naive_bayes_classifier(train_set)

# Evaluate the accuracy of the classifier on the test set
accuracy, confusion_matrix = evaluate_classifier(nb_classifier, test_set)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix)
