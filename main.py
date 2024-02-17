import os
import numpy as np
import tensorflow as tf
import pickle
import spacy
import en_core_web_lg
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from pymongo import MongoClient


# Load the trained model
model = load_model('next_word_model.h5')  # Update the path to your model

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Ensure you have the correct max_sequence_len used during training
max_sequence_len = 20  # Update this with the actual value used during training

def predict_next_words(text, n=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding='pre')
    predictions = model.predict(sequence, verbose=0)[0]

    top_indices = np.argsort(predictions)[-n:][::-1]
    top_words = []
    for i in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == i:
                top_words.append(word)
                break

    return top_words

# Load spaCy's English language model
nlp = en_core_web_lg.load()

# MongoDB connection URI
mongo_uri = os.getenv('MONGODB_URI') # Replace with your actual MongoDB URI
client = MongoClient(mongo_uri)

# Specify the database and collection
db = client.nextwordpredictiondb  # Database name
collection = db.vocabulary  # Collection name

def get_vocabulary_from_mongodb():
    vocabulary_docs = collection.find({})  # Fetch all documents
    vocabulary = [doc['word'] for doc in vocabulary_docs]  # Adjust the key if it's different in your documents
    return vocabulary

def find_synonyms(word, vocab, threshold=0.9):
    word = nlp(word)
    synonyms = []
    for v_word in vocab:
        v_word_spacy = nlp(v_word)
        if word.similarity(v_word_spacy) > threshold:
            synonyms.append(v_word)
    return synonyms

def predict_next_words_with_synonyms(text, n=3, threshold=0.55):
    vocabulary = get_vocabulary_from_mongodb()  # Fetch vocabulary from MongoDB
    top_words = predict_next_words(text, n)
    synonyms_in_vocab = {}

    for word in top_words:
        synonyms = find_synonyms(word, vocabulary, threshold)
        synonyms_in_vocab[word] = synonyms if synonyms else []

    return synonyms_in_vocab

def handle_prediction(predict_function, text, num_words):
    try:
        predictions = predict_function(text, num_words)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    data = request.json
    text = data['text']
    num_words = data.get('num_words', 3)
    return handle_prediction(predict_next_words, text, num_words)

@app.route('/predict-synonyms', methods=['GET'])
def predict_synonyms():
    data = request.json
    text = data['text']
    num_words = data.get('num_words', 3)
    return handle_prediction(predict_next_words_with_synonyms, text, num_words)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))