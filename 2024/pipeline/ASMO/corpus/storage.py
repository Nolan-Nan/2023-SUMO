"""
Stores corpus and trained classifiers.
"""

import pickle
import sklearn

def save_data(name, corpus):
    with open('ASMO/' + name + '/' + name + '_data.pkl', 'wb') as output:
      pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)

def load_data(name):
    # NOTE test here
    with open('ASMO/' + name + '/' + name + '_data.pkl', 'rb') as input:
      corpus = pickle.load(input)
    return corpus
