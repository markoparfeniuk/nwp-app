import numpy as np

from constants import SECONDS_IN_DAY
from dbClient.MongoDbClient import MongoDbClient
from models.logWordModel import LogWordModel

from datetime import datetime


class LearnWordsService:
    def __init__(self):
        self.mongoClient = MongoDbClient()
        self.logWordModel = LogWordModel()

    def get_main_vocabulary(self):
        result = self.mongoClient.get_main_vocabulary()

        return result

    def get_user_vocabulary(self, user_id):
        result = self.mongoClient.get_user_vocabulary(user_id)

        filtered_result = [item for item in result if not item.get('is_word_learnt', False)]

        return filtered_result

    def get_user_learning_vocabulary(self, user_id):
        result = self.mongoClient.get_user_learning_vocabulary(user_id)

        return result

    def prepare_words_for_log_model(self, words_list):
        current_time = datetime.now()

        result_array = []

        for obj in words_list:
            word = obj['word']

            word_from_vocabulary = self.mongoClient.get_main_word(word)

            aoa = word_from_vocabulary.get("Age_Of_Acquisition", None)
            freq = word_from_vocabulary.get("Log_Freq_HAL", None)
            con = word_from_vocabulary.get("Concreteness_Rating", None)

            time_seen = datetime.strptime(obj['time_seen'], "%Y-%m-%d %H:%M:%S")
            delta_time = current_time - time_seen

            history_wrong = obj['history_seen'] - obj['history_correct']

            x_delta_time = round(delta_time.total_seconds() / SECONDS_IN_DAY, 3)
            x_history_correct = int(obj['history_correct'])
            x_history_wrong = int(history_wrong)
            x_aoa = float(aoa)
            x_freq = float(freq)
            x_con = float(con)

            result_array.append({
                "word": word,
                "feature": [x_delta_time, x_history_correct, x_history_wrong, x_aoa, x_freq, x_con],
                "is forgotten": True,
                "probability": -1

            })

        return result_array

    def get_words_to_learn(self, words_array):
        feature_array = np.array([item["feature"] for item in words_array])

        my_predictions = self.logWordModel.predict_class(feature_array)

        for i, prediction in enumerate(my_predictions):
            words_array[i]["probability"] = prediction[0]
            words_array[i]["is forgotten"] = not bool(prediction[1])

        filtered_words_array = [item for item in words_array if item["is forgotten"]]

        return filtered_words_array

    def save_word_to_user_vocabulary(self, row, user_id):
        word_exists = self.mongoClient.check_if_word_exists_in_user_vocabulary(row['word'], user_id)

        if not word_exists:
            self.mongoClient.add_word_to_user_vocabulary(row, user_id)
            return True
        else:
            return False