import pickle
import numpy as np

from tqdm import tqdm

with open('../data/spelling_corpus_v2.pkl', 'rb') as f:
    inputs, targets = pickle.load(f)

input_chars = sorted(list(set(list(' '.join(inputs)))))
target_chars = sorted(list(set(list(' '.join(targets)))))

num_input_tokens = len(input_chars)
num_target_tokens = len(target_chars)

max_input_seq = max([len(text) for text in inputs])
max_target_seq = max([len(text) for text in targets])

print('Numer of Samples: {}'.format(len(inputs)))
print('Number of unique input tokens: {}'.format(num_input_tokens))
print('Number of unique target tokens: {}'.format(num_target_tokens))
print('Max sequence length for inputs: {}'.format(max_input_seq))
print('Max sequence length for targets: {}'.format(max_target_seq))

input_char2id = dict([(char, i) for i, char in enumerate(input_chars)])
input_id2char = dict([(i, char) for i, char in enumerate(input_chars)])

target_char2id = dict([(char, i) for i, char in enumerate(target_chars)])
target_id2char = dict([(i, char) for i, char in enumerate(target_chars)])

print('Creating Vectors')
input_vector = np.zeros((len(inputs), max_input_seq, num_input_tokens), dtype='float32')
target_vector = np.zeros((len(targets), max_target_seq, num_target_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(inputs, targets)):
    for t, char in enumerate(input_text):
        input_vector[i, t, input_char2id[char]] = 1.

    for t, char in enumerate(target_text):
        if t > 0:
            target_vector[i, t-1, target_char2id[char]] = 1.
        target_vector[i, t:, target_char2id[' ']] = 1.

with open('../data/input_char_tables_v2.pkl', 'wb') as f:
    pickle.dump((input_char2id, input_id2char), f)

with open('../data/target_char_tabesl_v2.pkl', 'wb') as f:
    pickle.dump((target_char2id, target_id2char), f)

with open('../data/vectorized_input_v2.pkl', 'wb') as f:
    pickle.dump((input_vector, target_vector), f, protocol=4)