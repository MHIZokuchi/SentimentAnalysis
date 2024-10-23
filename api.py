import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
with open('bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json(force=True)
    text = data['text']

    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])

    # Make predictions using the trained model
    prediction = model.predict(text_vectorized)[0]

    # Return the prediction as JSON
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(port=5000)