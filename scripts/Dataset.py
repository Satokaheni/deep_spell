import re
import string

from copy import copy

import numpy as np

from tqdm import tqdm


# Parameters for the model and dataset
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 ")

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
        self.input_data, self.target_data = self._generate_examples(data)

        self.input_data, self.target_data = self._vectorize()

    # preprocess the data
    def preprocess(self, x):
        # lowercase
        x = x.lower()

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
    def _generate_examples(self, dataset):
        print('Generating Examples...')

        inputs, targets = [], []

        self.input_characters = set()

        self.max_input_length = 0
        self.max_output_length = 0

        while dataset:
            data = dataset.pop()

            inputs.append(' '.join([self.add_spelling_errors(token) for token in data.split()]))
            targets.append('\t' + data + '\n')

            for char in inputs[-1]:
                self.input_characters.add(char)

            self.max_input_length = max([len(inputs[-1]), self.max_input_length])
            self.max_output_length = max([len(data[-1]), self.max_output_length])

        self.target_characters = copy(self.input_characters)
        self.target_characters.add('\t')
        self.target_characters.add('\n')

        self.char2id = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.num_input_tokens = len(self.input_characters)
        self.num_output_tokens = len(self.target_characters)

        return inputs, targets

    # vectorize data
    def _vectorize(self):
        input_vector = np.zeros((len(self.input_data), self.max_input_length, self.num_input_tokens), dtype='float32')
        output_vector = np.zeros((len(self.target_data), self.max_output_length, self.num_output_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_data, self.target_data)):
            for t, char in enumerate(input_text):
                input_vector[i, t, self.char2id[char]] = 1.
            input_vector[i, t + 1, self.char2id[' ']] = 1.

            for t, char in enumerate(target_text):
                if t > 0:
                    output_vector[i, t-1, self.char2id[char]] = 1.
                output_vector[i, t:, self.char2id[' ']] = 1.

        return input_vector, output_vector