import numpy as np
import pickle
import sys
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from transformers import AlbertTokenizer

# Load in data
print('Reading in data...')
with open('../data/bookcorpus/preprocess_p1.pkl', 'rb') as f:
    data = pickle.load(f)

# Corpus is too large randomly choose
random_choices = np.random.choice(len(data), size=int((10e5)*3), replace=False)

data = [data[random_choices[i]] for i in range(len(random_choices))]

# function to implement spelling errors
def add_spelling_errors(token, error_rate=.4):
    CHARS = list('abcdefghijklmnopqrstuvwxyz ')
    
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

# introduce spelling errors into the dataset
print('Creating Spelling Errors...')
spelling_errors = []
for i in tqdm(range(len(data))):
    spelling_errors.append(' '.join([add_spelling_errors(w) for w in data[i].split()]))

#Albert Tokenizer
tokenizer = AlbertTokenizer.from_pretrained('../data/albert')

# Get max length for padding
maxlen = max([len(sent) for sent in data])

print('Tokenizing the inputs...')
tokenized_input = []
for i in tqdm(range(len(spelling_errors))):
    tokenized_input.append(tokenizer.encode(spelling_errors[i]))

print('Tokenizing the targets...')
tokenized_target = []
for i in tqdm(range(len(data))):
    tokenized_target.append(tokenizer.encode(data[i]))

print('Padding The Data...')
tokenized_input = pad_sequences(tokenized_input, maxlen=maxlen, dtype='long', truncating='post', padding='post')
tokenized_target = pad_sequences(tokenized_target, maxlen=maxlen, dtype='long', truncating='post', padding='post')

print('Splitting into train and val...')
train_inputs, val_inputs, train_targets, val_targets = train_test_split(tokenized_input, tokenized_target, random_state=3389, test_size=.2)

print('Saving Data...')
with open('../data/train_data.pkl', 'wb') as f:
    train_data = (train_inputs, train_targets)
    pickle.dump(train_data, f)

with open('../data/val_data.pkl', 'wb') as f:
    val_data = (val_inputs, val_targets)
    pickle.dump(val_data, f)