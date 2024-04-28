import os
import random

from datetime import datetime
from flask import Flask, request, jsonify

from services.LearnWordsService import LearnWordsService
from services.PredictWordsService import PredictWordsService

learnWordsService = LearnWordsService()
predictWordsService = PredictWordsService()

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    data = request.json
    user_id = data['user_id']
    text = data['text']
    num_words = data.get('num_words', 3)
    try:
        predictions = predictWordsService.predict_next_words(user_id, text, num_words)
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


@app.route('/learn', methods=['GET'])
def get_new_word_to_learn():
    user_id = request.args.get('user_id')

    main_vocabulary = learnWordsService.get_main_vocabulary()

    # ADD HOW TO CHOOSE WORD

    random_word = random.choice(main_vocabulary)

    time_seen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {'word': random_word['Word'], 'time_seen': time_seen, 'history_seen': 1, 'history_correct': 0,
               "is_word_learnt": False}

    is_word_added = learnWordsService.save_word_to_user_vocabulary(new_row, user_id)

    if is_word_added:
        return jsonify({"message": f"Word '{new_row['word']}' was added to user vocabulary successfully!"}), 200
    else:
        return jsonify({
            "error": f"Word '{new_row['word']}' already exists in user vocabulary."}), 404


@app.route('/relearn', methods=['GET'])
def get_words_to_relearn():
    user_id = request.args.get('user_id')

    # DECIDE HOW MANY WORDS WILL BE SENT FOR RELEARNING

    user_vocabulary = learnWordsService.get_user_learning_vocabulary(user_id)

    if not user_vocabulary:
        return jsonify({'error': 'User has no words to relearn'}), 404

    words_for_log_model = learnWordsService.prepare_words_for_log_model(user_vocabulary)
    words_to_learn = learnWordsService.get_words_to_learn(words_for_log_model)

    if not words_to_learn:
        return jsonify({'error': 'User has no words to relearn. All words have high probability'}), 404

    sorted_data = sorted(words_to_learn, key=lambda x: x['probability'], reverse=False)
    sorted_word_array = [{'word': d['word']} for d in sorted_data]

    response = {'words': sorted_word_array}

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))