import numpy as np

import utils


class Baseline(object):
    """
    The baseline model.
    """

    def __init__(self, pos_tags, words, training_set):
        """
        The init function of the Baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        """
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}

        self.multinomial_prob, self.emission_prob = baseline_mle(training_set, self)

    def MAP(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        """
        result = []
        for words in sentences:
            possible_tags = list()
            for word in words:
                max_index = np.argmax(self.multinomial_prob * self.emission_prob.T[self.word2i[word]])
                possible_tags.append(self.pos_tags[max_index])
            result.append(possible_tags)
        return result


def baseline_mle(training_set, model):
    """
    Calculates the Maximum Likelihood estimation of the multinomial and emission probabilities for the baseline model.

    :param training_set: an iterable sequence of sentences, each containing both the words and the PoS tags of the
            sentence.
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities.
    """
    multinomial_probs = np.zeros(model.pos_size)
    emission_prob = np.zeros((model.pos_size, model.words_size))

    for row in training_set:
        pos_tags = row[0]
        words = row[1]

        for i in range(len(pos_tags)):
            emission_prob[model.pos2i[pos_tags[i]], model.word2i[words[i]]] += 1
            multinomial_probs[model.pos2i[pos_tags[i]]] += 1

    return multinomial_probs / np.sum(multinomial_probs), utils.normalize_prob(emission_prob)
