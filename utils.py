import numpy as np

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'


def normalize_prob(prob):
    normalization = np.sum(prob, axis=1)
    normalization = normalization.reshape((normalization.shape[0], 1))
    return np.divide(prob, normalization, out=np.zeros_like(prob), where=normalization != 0)


def process_data_set(data_set):
    return [(_pos_add_states(data_set[i][0]), _word_add_states(data_set[i][1])) for i in range(len(data_set))]


def process_words_pos(words, pos):
    return _word_add_states(words), _pos_add_states(pos)


def handle_rare_words(words, data, n_times):
    word_dict = dict()
    for row in data:
        for word in row[1]:
            if word not in word_dict.keys():
                word_dict[word] = 0
            word_dict[word] += 1

    words = [word for word in words if word_dict[word] > n_times] + [RARE_WORD]
    data = [_add_rare_word_to_row(row, word_dict, n_times) for row in data]

    return words, data


def _pos_add_states(data):
    return [START_STATE] + data + [END_STATE]


def _word_add_states(data):
    return [START_WORD] + data + [END_WORD]


def _add_rare_word_to_row(row, word_dict, n_times):
    for i in range(len(row[1])):
        if word_dict[row[1][i]] <= n_times:
            row[1][i] = RARE_WORD
    return row
