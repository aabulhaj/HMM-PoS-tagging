import numpy as np

import utils


class HMM(object):
    """
    The basic HMM_Model with multinomial transition functions.
    """

    def __init__(self, pos_tags, words, training_set):
        """
        The init function of the basic HMM Model.
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

        self.transition_prob, self.emission_prob = hmm_mle(training_set, self)

    def viterbi(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        """
        min_transition = np.min(self.transition_prob[self.transition_prob > 0]) / 2
        min_emission = np.min(self.emission_prob[self.emission_prob > 0]) / 2

        transition_prob = self.transition_prob
        emission_prob = self.emission_prob

        transition_prob[transition_prob == 0] = min_transition
        emission_prob[emission_prob == 0] = min_emission

        transition_prob = np.log(transition_prob)
        emission_prob = np.log(emission_prob).T

        result = []
        for words in sentences:
            num_of_words = len(words)

            # Initialize empty table
            T1 = np.zeros((num_of_words, self.pos_size))
            T2 = np.zeros_like(T1)

            # Base case Pr(j|START)*Pr(x_1|y_1=j)
            T1[0] = transition_prob[self.pos2i[utils.START_STATE]] + emission_prob[self.word2i[words[0]]]

            for t in range(1, num_of_words):
                vals = [T1[t - 1] + transition_prob.T[i] + emission_prob[self.word2i[words[t]]][i]
                        for i in range(self.pos_size)]
                T1[t] = np.max(vals, axis=1)
                T2[t] = np.argmax(vals, axis=1)

            z = np.zeros(num_of_words)
            x = [''] * num_of_words

            z[-1] = np.argmax(T1[-1])
            x[-1] = self.pos_tags[int(z[-1])]

            for i in range(len(words) - 1, 0, -1):
                z[i - 1] = T2[i, int(z[i])]
                x[i - 1] = self.pos_tags[int(z[i - 1])]

            result.append(x)

        return result


def hmm_mle(training_set, model):
    """
    Calculates the Maximum Likelihood estimation of the transition and emission probabilities for the standard
        multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing both the words and the PoS tags
            of the sentence.
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities.
    """
    transition_prob = np.zeros((model.pos_size, model.pos_size))
    emission_prob = np.zeros((model.pos_size, model.words_size))

    for row in training_set:
        pos_tags = row[0]
        words = row[1]

        for i in range(len(pos_tags) - 1):
            transition_prob[model.pos2i[pos_tags[i]], model.pos2i[pos_tags[i + 1]]] += 1
            emission_prob[model.pos2i[pos_tags[i]], model.word2i[words[i]]] += 1
        emission_prob[model.pos2i[pos_tags[-1]], model.word2i[words[-1]]] += 1

    return utils.normalize_prob(transition_prob), utils.normalize_prob(emission_prob)
