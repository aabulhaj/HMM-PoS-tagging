import pickle

import baseline
import hmm
import utils


def evaluate(pred_func, test_set, print_stats=True):
    predictions = pred_func([row[1][1:-1] for row in test_set])

    total_predictions = 0
    correct_predictions = 0
    for i in range(len(predictions)):
        actual_val = test_set[i][0]
        prediction = predictions[i]
        for j in range(len(prediction)):
            total_predictions += 1
            if prediction[j] == actual_val[1:-1][j]:
                correct_predictions += 1

    if print_stats:
        print('\nEvaluating on total of: {}'.format(total_predictions))
        print('Correct predictions: {}'.format(correct_predictions))
        print('Correct predictions percentage: {}'.format(correct_predictions / total_predictions))

    return total_predictions, correct_predictions


if __name__ == '__main__':
    # Load data.
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    # Clean data.
    data = utils.process_data_set(data)
    words, data = utils.handle_rare_words(words, data, 2)
    words, pos = utils.process_words_pos(words, pos)

    # Initialize training and testing set.
    training_set_size = int(0.8 * len(data))
    training_set = data[: training_set_size]
    test_set = data[training_set_size:]

    # Initialize and evaluate the baseline model.
    baseline = baseline.Baseline(pos, words, training_set)
    total_predictions, correct_predictions = evaluate(baseline.MAP, test_set)

    # Find the models accuracy.
    accuracy = (correct_predictions / total_predictions) * 100
    print('Baseline model accuracy: {}'.format(accuracy))

    # Initialize and evaluate the HMM model.
    hmm_model = hmm.HMM(pos, words, training_set)
    total_predictions, correct_predictions = evaluate(hmm_model.viterbi, test_set)

    # Find the models accuracy.
    accuracy = (correct_predictions / total_predictions) * 100
    print('HMM model accuracy: {}'.format(accuracy))
