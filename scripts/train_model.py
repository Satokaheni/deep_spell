from utils import create_training_data
from model import build_model

import pickle

# Read in sentences
print('Reading in Data...')
with open('data/english_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

print('Creating Data for model...')
x, y, encoding_dict, decoding_dict = create_training_data(sentences)

with open('data/training_data.pkl', 'wb') as f:
    pickle.dump({'x': x, 'y': y, 'encode_dict': encoding_dict, 'decode_dict': decoding_dict}, f)