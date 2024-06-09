import os
import numpy as np
import pickle
import en_core_web_lg

from constants import MAX_SEQUENCE_LENGTH
from dbClient.MongoDbClient import MongoDbClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

 
class PredictWordsService:
    def __init__(self):
        self.mongoClient = MongoDbClient()
        self.langModel = load_model('/home/site/wwwroot/saved_models/next_word_model.h5')
        self.nlp = en_core_web_lg.load()
        with open('/home/site/wwwroot/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def get_user_learning_vocabulary(self, user_id):
        result = self.mongoClient.get_user_vocabulary(user_id)

        filtered_result = [item['word'] for item in result if not item.get('is_word_learnt', False)]

        return filtered_result

    def get_user_vocabulary(self, user_id):
        result = self.mongoClient.get_user_vocabulary(user_id)

        return result

    def predict_next_words(self, text, n=3):
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_SEQUENCE_LENGTH - 1, padding='pre')
        predictions = self.langModel.predict(sequence, verbose=0)[0]

        top_indices = np.argsort(predictions)[-n:][::-1]
        top_words = []
        for i in top_indices:
            for word, index in self.tokenizer.word_index.items():
                if index == i:
                    top_words.append(word)
                    break

        return top_words

    def find_synonyms(self, word, vocab, threshold=0.9):
        word_doc = self.nlp(word)
        exclude_tags = {'DT', 'PRP', 'IN', 'CC', 'WDT', 'NNP'}

        if word_doc[0].tag_ in exclude_tags:
            return []
        synonyms = []
        for v_word in vocab:
            v_word_doc = self.nlp(v_word)
            if v_word_doc[0].tag_ not in exclude_tags:
                similarity = word_doc.similarity(v_word_doc)
                if similarity > threshold:
                    synonyms.append((v_word, similarity))
        synonyms = sorted(synonyms, key=lambda x: x[1], reverse=True)[:3]
        return [synonym[0] for synonym in synonyms]

    def predict_next_words_with_synonyms(self, user_id, text, n=3, threshold=0.55):
        vocabulary = self.get_user_learning_vocabulary(user_id)
        top_words = self.predict_next_words(text, n)
        synonyms_in_vocab = {}

        for word in top_words:
            synonyms = self.find_synonyms(word, vocabulary, threshold)
            synonyms_in_vocab[word] = [] if not synonyms or word == 'statement' else synonyms

        return synonyms_in_vocab
