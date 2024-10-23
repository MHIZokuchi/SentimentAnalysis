from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
app = Flask(__name__)
# Load the saved model
model = load_model('sentiment_analysis_model.h5')
# Load the saved tokenizer
with open('word_tokenizer.pkl', 'rb') as tokenizer_file:
    word_tokenizer = pickle.load(tokenizer_file)
maxlen = 100
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data['text']

    # Preprocess the input text
    sequence = word_tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, padding='post', maxlen=maxlen)

    # Make predictions
    predictions = model.predict(sequence)
    predicted_class = np.argmax(predictions)

    # Map the predicted class to sentiment label
    sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    #sentiment_labels = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}

    predicted_sentiment = sentiment_labels[predicted_class]
    # Return the predicted sentiment
    return jsonify({'sentiment': predicted_sentiment})
if __name__ == '__main__':
    app.run(port=5000)
