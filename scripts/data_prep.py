from utils import create_training_data
from model import build_model

import pickle

# Read in sentences
print('Reading in Data...')
with open('data/english_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

print('Creating Data for model...')
encoder_data, decoder_data, encoding_dict = create_training_data(sentences)

with open('data/encoding_dict.pkl', 'wb') as f:
    pickle.dump(encoding_dict, f, protocol=4)

with open('data/encode_data.pkl', 'wb') as f:
    pickle.dump(encoder_data, f, protocol=4)

with open('data/target_data.pkl', 'wb') as f:
    pickle.dump(decoder_data, f, protocol=4)