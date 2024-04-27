import os
from pymongo import MongoClient


class MongoDbClient:
    def __init__(self):
        connection_string = os.getenv('MONGODB_URI')
        db_name = 'word_app'
        main_vocabulary_collection_name = 'vocabulary'
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
