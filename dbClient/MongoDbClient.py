import os
from pymongo import MongoClient
from datetime import datetime


class MongoDbClient:
    def __init__(self):
        connection_string = os.getenv('MONGODB_URI')
        db_name = 'word_app_db'
        main_vocabulary_collection_name = 'main_vocabulary'
        users_vocabulary_collection_name = 'users_vocabulary'
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.main_vocabulary_collection = self.db[main_vocabulary_collection_name]
        self.users_vocabulary_collection = self.db[users_vocabulary_collection_name]

    def get_main_vocabulary(self):
        result = list(self.main_vocabulary_collection.find({}))

        return result

    def get_main_word(self, word):
        result = self.main_vocabulary_collection.find_one({"Word": word})

        return result

    def get_user_vocabulary(self, user_id):
        result = self.users_vocabulary_collection.find_one({'_user_id': user_id})

        return result.get('vocabulary', []) if result else []

    def get_user_level(self, user_id):
        user_doc = self.users_vocabulary_collection.find_one({'_user_id': user_id})

        if user_doc:
            return user_doc.get('level', None)
        else:
            return None

    def get_user_learning_vocabulary(self, user_id):
        pipeline = [
            {"$match": {"_user_id": user_id}},
            {"$project": {
                "vocabulary": {
                    "$filter": {
                        "input": "$vocabulary",
                        "as": "word",
                        "cond": {"$eq": ["$$word.is_word_learnt", False]}
                    }
                },
                "_id": 0
            }}
        ]

        user_vocabulary = self.users_vocabulary_collection.aggregate(pipeline)

        if user_vocabulary:
            if user_vocabulary.alive:
                return list(user_vocabulary)[0]['vocabulary']
        return []

    def check_if_word_exists_in_user_vocabulary(self, word, user_id):
        query = {
            '_user_id': user_id,
            'vocabulary.word': word
        }
        result = self.users_vocabulary_collection.find_one(query)

        return result is not None

    def add_word_to_user_vocabulary(self, row, user_id):
        self.users_vocabulary_collection.update_one({"_user_id": user_id}, {"$push": {"vocabulary": row}}, upsert=True)

    def set_user_level(self, user_id, level):
        doc = self.users_vocabulary_collection.find_one({"_user_id": user_id})
        if doc:
            self.users_vocabulary_collection.update_one({"_id": doc["_id"]}, {"$set": {"level": level}})
        else:
            new_document = {"_user_id": user_id, "level": level}
            self.users_vocabulary_collection.insert_one(new_document)

    def add_words_array_to_user_vocabulary(self, user_id, words):
        result = self.users_vocabulary_collection.update_one(
            {"_user_id": user_id}, {"$push": {"vocabulary": {"$each": words}}})

        if result.modified_count > 0:
            return True
        else:
            return False

    def get_words_by_level(self, level):
        result = self.main_vocabulary_collection.find({"level": level})

        return result

    def update_user_vocabulary_word(self, user_id, word, repetition_result, is_word_learnt):
        new_time_seen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        filter_criteria = {
            "_user_id": user_id,
            "vocabulary.word": word
        }

        update_operation = {
            "$set": {
                "vocabulary.$.time_seen": new_time_seen,
                "vocabulary.$.is_word_learnt": is_word_learnt
            },
            "$inc": {
                "vocabulary.$.history_seen": 1,
                "vocabulary.$.history_correct": int(repetition_result)
            }
        }

        update_result = self.users_vocabulary_collection.find_one_and_update(
            filter_criteria,
            update_operation,
        )

        if update_result is not None:
            return True
        else:
            return False

    def get_user_vocabulary_word(self, user_id, word):
        result = self.users_vocabulary_collection.find_one({"_user_id": user_id, "vocabulary.word": word})

        return result

    def increment_word_history_seen(self, user_id, word):
        result = self.users_vocabulary_collection.update_one(
            {
                '_user_id': user_id,
                'vocabulary.word': word
            },
            {
                '$inc': {'vocabulary.$.history_seen': 1}
            }
        )

        if result.matched_count == 0:
            return -1

        return 0

    def update_one(self, filter, update):
        try:
            self.db.user_vocabulary.update_one(filter, update)
        except Exception as e:
            print(f"Error updating document: {e}")