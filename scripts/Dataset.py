from collections import Counter
import re
import string
from time import time

import numpy as np
from numpy.random import choice as random_choice
from numpy.random import randint as random_randint
from numpy.random import shuffle as random_shuffle
from numpy.random import rand
from numpy import zeros as np_zeros 

from tqdm import tqdm

from contractions import contractions

# Parameters for the model and dataset
MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 3
AMOUNT_OF_NOISE = 0.2 / MAX_SENTENCE_LENGTH
NUMBER_OF_CHARS = 100  # 75
CHARS = list("abcdefghijklmnopqrstuvwxyz ")

# Regular Expression replacement to regularize text
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
                                        chr(768), chr(769), chr(832), chr(833), chr(2387), chr(5151),
                                        chr(5152), chr(65344), chr(8242)),
                                    re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)

# Dataset Class to handle creating the dataset
class DataSet:
    def __init__(self, dataset_filename, test_set_fraction=.2, inverted=True):
        self.inverted = inverted

        # read in data
        data = self.read_dataset(dataset_filename)

        # generate input and targets
        inputs, targets = self.generate_examples(data)

    # preprocess the data
    def preprocess(self, x):
        # lowercase
        x = x.lower()

        # replace contractions
        # x = ' '.join([contractions[w] if w in contractions else w for x in x.split()])

        # Normalize parenthesis, dashes, and apostrophies
        x = RE_DASH_FILTER.sub('-', x)
        x = RE_APOSTROPHE_FILTER.sub("'", x)
        x = RE_LEFT_PARENTH_FILTER.sub("(", x)
        x = RE_RIGHT_PARENTH_FILTER.sub(")", x)

        # remove punctuation except apostrophies
        x = x.translate(str.maketrans('', '', string.punctuation.replace("'", '')))

        return x


    # Read in dataset
    def read_dataset(self, filename):
        print('Reading Dataset...')
        with open(filename, 'r') as f:
            data = f.readlines()

        print('Read in {} lines of data from corpus'.format(len(data)))

        print('Preprocessing Data..')
        for i in tqdm(range(len(data))):
            data[i] = self.preprocess(data[i])

        return data


    # Create Random Spelling Error
    def add_spelling_errors(self, token, error_rate=.4):
        """Simulate some artificial spelling mistakes."""
        assert(0.0 <= error_rate < 1.0)
        if len(token) < 3:
            return token
        rand = np.random.rand()
        # Here are 4 different ways spelling mistakes can occur,
        # each of which has equal chance.
        prob = error_rate / 4.0
        if rand < prob:
            # Replace a character with a random character.
            random_char_index = np.random.randint(len(token))
            token = token[:random_char_index] + np.random.choice(CHARS) \
                    + token[random_char_index + 1:]
        elif prob < rand < prob * 2:
            # Delete a character.
            random_char_index = np.random.randint(len(token))
            token = token[:random_char_index] + token[random_char_index + 1:]
        elif prob * 2 < rand < prob * 3:
            # Add a random character.
            random_char_index = np.random.randint(len(token))
            token = token[:random_char_index] + np.random.choice(CHARS) \
                    + token[random_char_index:]
        elif prob * 3 < rand < prob * 4:
            # Transpose 2 characters.
            random_char_index = np.random.randint(len(token) - 1)
            token = token[:random_char_index]  + token[random_char_index + 1] \
                    + token[random_char_index] + token[random_char_index + 2:]
        else:
            # No spelling errors.
            pass
        return token
    
    # Generate sentences with mispellings
    def generate_examples(self, dataset):
        print('Generating Examples...')

        inputs, targets = [], []

        while dataset:
            data = dataset.pop()

            inputs.append(' '.join([self.add_spelling_errors(token) for token in data.split()]))
            targets.append(data)

            
            