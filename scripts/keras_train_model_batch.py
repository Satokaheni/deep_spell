from model import build_model
from utils import create_model_data

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

import numpy as np
import pickle

import sys

sample_size = int(sys.argv[1])
i = int(sys.argv[2])

# Read in data
print('Reading in Data...')
with open('data/encoding_dict.pkl', 'rb') as f:
    encoding_dict = pickle.load(f)

with open('data/encode_data.pkl', 'rb') as f:
    encoder_data = pickle.load(f)[sample_size*i:sample_size*(i+1)]

with open('data/target_data.pkl', 'rb') as f:
    decoder_data = pickle.load(f)[sample_size*i:sample_size*(i+1)]


max_input_len = encoder_data.shape[1]
input_vocab_size = len(encoding_dict['w2id'].keys()) + 2
max_output_len = decoder_data.shape[1]
output_vocab_size = len(encoding_dict['w2id'].keys()) + 2

if i == 0:
    print('Building Model...')
    model = build_model(max_input_len, input_vocab_size, max_output_len, output_vocab_size)
    model.summary()

else:
    print('Loading Previous Best Model...')
    model = load_model('data/model_checkpoint.h5')

checkpoint = ModelCheckpoint('data/model_checkpoint.h5', monitor='val_accuracy', save_best_only=True)
earlystop = EarlyStopping(monitor='val_accuracy', patience=20)
callbacks = [checkpoint, earlystop]

x, y = create_model_data(encoder_data, decoder_data, output_vocab_size)

print('Training Model on new sample data...')
model.fit(x, y, batch_size=64, epochs=50, validation_split=.2, callbacks=callbacks)