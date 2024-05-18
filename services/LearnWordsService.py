import numpy as np

from constants import SECONDS_IN_DAY, MIN_DELTA_TIME, LEVEL_ORDER, HISTORY_CORRECT_THRESHOLD
from dbClient.MongoDbClient import MongoDbClient
from models.logWordModel import LogWordModel

from datetime import datetime, timedelta


class LearnWordsService:
    def __init__(self):
        self.mongoClient = MongoDbClient()
        self.logWordModel = LogWordModel()

    def get_main_vocabulary(self):
        result = self.mongoClient.get_main_vocabulary()

        return result

    def get_user_vocabulary(self, user_id):
        result = self.mongoClient.get_user_vocabulary(user_id)

        return result

    def get_user_learning_vocabulary(self, user_id):
        result = self.mongoClient.get_user_learning_vocabulary(user_id)

        filtered_result = [item for item in result if not item.get('is_word_learnt', False)]

        return filtered_result

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

            result_array = [item for item in result_array if item["feature"][0] >= MIN_DELTA_TIME]

        return result_array

    def get_words_to_learn(self, words_array):
        feature_array = np.array([item["feature"] for item in words_array])

        my_predictions = self.logWordModel.predict_class(feature_array)

        for i, prediction in enumerate(my_predictions):
            words_array[i]["probability"] = prediction[0]
            words_array[i]["is forgotten"] = not bool(prediction[1])

        filtered_words_array = [item for item in words_array if item["is forgotten"]]

        return filtered_words_array

    def check_if_word_exists_in_user_vocabulary(self, word, user_id):
        result = self.mongoClient.check_if_word_exists_in_user_vocabulary(word, user_id)

        return result

    def save_word_to_user_vocabulary(self, row, user_id):
        word_exists = self.check_if_word_exists_in_user_vocabulary(row['word'], user_id)

        if not word_exists:
            self.mongoClient.add_word_to_user_vocabulary(row, user_id)
            return True
        else:
            return False

    def get_word_definition(self, word):
        word_doc = self.mongoClient.get_main_word(word)

        if word_doc and "Definitions" in word_doc:
            return word_doc["Definitions"]
        else:
            return None

    def get_word_level(self, word):
        word_doc = self.mongoClient.get_main_word(word)

        if word_doc and "level" in word_doc:
            return word_doc["level"]
        else:
            return None

    def add_words_to_user_vocabulary(self, user_id, words):
        words_to_add = [{"word": word["Word"], "is_word_learnt": True} for word in words]
        self.mongoClient.add_words_array_to_user_vocabulary(user_id, words_to_add)

    def get_words_by_level(self, level):
        result = self.mongoClient.get_words_by_level(level)

        return result

    def set_user_level(self, user_id, level):
        self.mongoClient.set_user_level(user_id, level)

        user_level_index = LEVEL_ORDER.index(level)

        for i in range(user_level_index):
            current_level = LEVEL_ORDER[i]
            words_for_current_level = self.get_words_by_level(current_level)
            self.add_words_to_user_vocabulary(user_id, words_for_current_level)

    def get_user_level(self, user_id):
        result = self.mongoClient.get_user_level(user_id)

        return result

    def handle_repetition_result(self, user_id, word, repetition_result):
        is_word_learnt = False
        successful_result = False

        user_vocabulary = self.mongoClient.get_user_learning_vocabulary(user_id)

        if user_vocabulary:
            user_word = None
            for entry in user_vocabulary:
                if entry['word'] == word:
                    user_word = entry
                    break

            if user_word:
                history_correct = user_word.get('history_correct', 0)
                time_seen = user_word.get('time_seen')
                time_seen_datetime = datetime.strptime(time_seen, "%Y-%m-%d %H:%M:%S")

                if (repetition_result and (history_correct > (HISTORY_CORRECT_THRESHOLD - 1))
                        and (datetime.now() - time_seen_datetime > timedelta(days=HISTORY_CORRECT_THRESHOLD))):
                    is_word_learnt = True

                successful_result = self.mongoClient.update_user_vocabulary_word(user_id, word, repetition_result,
                                                                                 is_word_learnt)

        return successful_result

    def increment_word_history_seen(self, user_id, word):
        result = self.mongoClient.increment_word_history_seen(user_id, word)

        return result

    def update_word_status(self, user_id, word, new_status):
        try:
            self.mongoClient.update_one(
                {'user_id': user_id, 'word': word},
                {'$set': {'is_word_learnt': new_status}}
            )
            return True
        except Exception as e:
            print(f"Error updating word status: {e}")
            return False