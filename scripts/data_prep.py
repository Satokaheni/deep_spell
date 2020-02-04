import numpy
import string
import pickle

import spacy

import numpy as np

from tqdm import tqdm

NUM_MISSPELLED_COPIES = 1

CHARS = list('abcdefghijklmnopqrstuvwxyz ')
REMOVE_PUNC = string.punctuation.replace("'", '').replace(',', '') 


# read in data
print('Reading in Data')
with open('../data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)


# preprocess each string
def preprocess(x):
    # remove tabs and newlines
    x = x.rstrip()
    
    # lowercase
    x = x.lower()

    # remove punc but keep apostrophe
    x = x.translate(str.maketrans('', '', REMOVE_PUNC))

    return x

print('Preprocessing')
for i in tqdm(range(len(corpus))):
    corpus[i] = preprocess(corpus[i])


max_sent_len = max([len(sent) for sent in corpus])
print('Max Sentence Length: {}'.format(max_sent_len))


def add_spelling_errors(token, error_rate):
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

print('Creating Spelling Errors')
inputs = []
targets = []
for i in tqdm(range(len(corpus))):
    # Add different misspellings to same target
    for _ in range(NUM_MISSPELLED_COPIES):
        inputs.append(' '.join([add_spelling_errors(token, .3) for token in corpus[i].split()]))
        targets.append('\t' + corpus[i] + '\n')

    # Also add in target to target so spelled correct input decodes to same output
    inputs.append(corpus[i])
    targets.append('\t' + corpus[i] + '\n')

data = (inputs, targets)

with open('../data/spelling_corpus_v2.pkl', 'wb') as f:
    pickle.dump(data, f)
