# Processing the data
The supplied pickle files are not fully processed. Helper functions that add START and END hidden states (PoS tags) are implemented in utils.

Rare words were handled as follows: We remove words that appear more than n times (I use n=2) from our words list and then replace each occurance of a rare word in the dataset by RARE.

An example of how I clean the pickles can be found in models_evaluator.py


# Baseline model
To see how the HMM model improves upon the most naive model, I implemented a baseline model. In this model, we will assume that the hidden states are actually independent. This means that we only need to learn the emission probabilities and multinomial probabilities over the hidden states.


# HMM
Implemented Maximum Likelihood estimator for the standard multinomial HMM.
[Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) was implemented in order to calculate the MAP assignment of PoS tags for a given sentence.


# Results
I evaluate both models on a test set that contains 221932 tags.
The baseline model accuracy was 90.85% and the HMM accuracy was 94.14%.


# Running example
models_evaluator.py contains a full example that loads the pickles, processes the data as described above, builds a baseline model and a HMM, evaluates them and prints the results.
