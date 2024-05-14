import os
import random

from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from constants import WORDS_AMOUNT_TO_RELEARN
from services.LearnWordsService import LearnWordsService
from services.PredictWordsService import PredictWordsService

learnWordsService = LearnWordsService()
predictWordsService = PredictWordsService()

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['GET'])
def predict():
    data = request.json
    text = data['text']
    num_words = data.get('num_words', 3)
    try:
        predictions = predictWordsService.predict_next_words(text, num_words)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict-synonyms', methods=['GET'])
def predict_synonyms():
    data = request.json
    user_id = data['user_id']
    text = data['text']
    num_words = data.get('num_words', 3)
    try:
        predictions = predictWordsService.predict_next_words_with_synonyms(user_id, text, num_words)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/set_user_level', methods=['POST'])
def set_level():
    request_data = request.get_json()
    user_id = request_data.get('user_id')
    level = request_data.get('level')

    learnWordsService.set_user_level(user_id, level)

    return jsonify({'message': 'User level set successfully'}), 200


@app.route('/get_random_word', methods=['GET'])
def get_new_word_to_learn():
    user_id = request.args.get('user_id')

    user_level = learnWordsService.get_user_level(user_id)
    level_vocabulary = learnWordsService.get_words_by_level(user_level)
    user_vocabulary = learnWordsService.get_user_vocabulary(user_level)

    user_words_set = set(word['word'] for word in user_vocabulary)
    level_vocabulary_set = set(word['Word'] for word in level_vocabulary)

    filtered_level_vocabulary = [word for word in level_vocabulary_set if word not in user_words_set]

    if filtered_level_vocabulary:
        random_word = random.choice(filtered_level_vocabulary)

        word_exists = learnWordsService.check_if_word_exists_in_user_vocabulary(random_word, user_id)

        if not word_exists:
            definition = learnWordsService.get_word_definition(random_word)
            return jsonify({"word": random_word, "definition": definition}), 200
        else:
            return jsonify({"error": f"Word '{random_word}' already exists in user vocabulary."}), 404
    else:
        return jsonify({"error": f"There are no words to learn"}), 404


@app.route('/learn_word', methods=['POST'])
def learn_word():
    request_data = request.get_json()
    user_id = request_data.get('user_id')
    word = request_data.get('word')

    time_seen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {'word': word, 'time_seen': time_seen, 'history_seen': 1, 'history_correct': 0,
               "is_word_learnt": False}

    is_word_added = learnWordsService.save_word_to_user_vocabulary(new_row, user_id)

    if is_word_added:
        definition = learnWordsService.get_word_definition(new_row['word'])
        return jsonify({"word": new_row['word'], "definition": definition}), 200
    else:
        return jsonify({"error": f"Word '{new_row['word']}' already exists in user vocabulary."}), 404


@app.route('/get_words_to_relearn', methods=['GET'])
def get_words_to_relearn():
    user_id = request.args.get('user_id')

    user_vocabulary = learnWordsService.get_user_learning_vocabulary(user_id)

    if not user_vocabulary:
        return jsonify({'massage': 'User has no words to relearn'}), 200

    words_for_log_model = learnWordsService.prepare_words_for_log_model(user_vocabulary)

    if len(words_for_log_model) < 1:
        return jsonify({'massage': 'User has no words to relearn'}), 200

    words_to_learn = learnWordsService.get_words_to_learn(words_for_log_model)

    if not words_to_learn:
        return jsonify({'massage': 'User has no words to relearn. All words have high probability'}), 200

    sorted_data = sorted(words_to_learn, key=lambda x: x['probability'], reverse=False)

    sorted_word_array = [{'word': d['word']} for d in sorted_data]

    if len(sorted_word_array) <= WORDS_AMOUNT_TO_RELEARN:
        result_array = sorted_word_array
    else:
        result_array = sorted_word_array[:WORDS_AMOUNT_TO_RELEARN]

    for item in result_array:
        word = item["word"]
        definition = learnWordsService.get_word_definition(word)
        item["definition"] = definition

    response = {'words': result_array}

    return jsonify(response), 200


@app.route('/relearn_result', methods=['POST'])
def handle_word_relearn_result():
    request_data = request.get_json()
    user_id = request_data.get('user_id')
    word = request_data.get('word')
    repetition_result = request_data.get('result')

    successful_result = learnWordsService.handle_repetition_result(user_id, word, repetition_result)

    if successful_result:
        return jsonify({'message': 'Result of repetition was saved successfully'}), 200
    else:
        return jsonify({'error': 'Error occurred while saving the repetition result'}), 404


@app.route('/set_word_as_known', methods=['POST'])
def set_word_as_known():
    request_data = request.get_json()
    user_id = request_data.get('user_id')
    word = request_data.get('word')

    new_row = {'word': word, "is_word_learnt": True}

    is_word_added = learnWordsService.save_word_to_user_vocabulary(new_row, user_id)

    if is_word_added:
        return jsonify({'message': 'Word was saved as known'}), 200
    else:
        return jsonify({"error": f"Word '{new_row['word']}' already exists in user vocabulary."}), 404


@app.route('/get_user_vocabulary', methods=['GET'])
def get_user_vocabulary():
    user_id = request.args.get('user_id')

    user_vocabulary = learnWordsService.get_user_vocabulary(user_id)

    result = [entry['word'] for entry in user_vocabulary]

    response = {'user_vocabulary': result}

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
