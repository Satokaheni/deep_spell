from keras.models import Model, Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Adamax, RMSprop

import numpy as np

import pickle

batch_size = 64
epochs = 100
latent_dim = 256
num_encoder_tokens = 73
num_decoder_tokens = 75
layer_numbers = 2
max_target_len = 260


with open('../data/input_char_tables_v2.pkl', 'rb') as f:
    input_char2id, input_id2char = pickle.load(f)

with open('../data/target_char_tabesl_v2.pkl', 'rb') as f:
    target_char2id, target_id2char = pickle.load(f)

with open('../data/vectorized_input_v2.pkl', 'rb') as f:
    input_vector, target_vector = pickle.load(f)

model = Sequential()

for _ in range(layer_numbers-1):
    model.add(layers.LSTM(latent_dim, input_shape=(None, num_encoder_tokens), init='he_normal', return_sequences=True))
    model.add(layers.Dropout(.5))
model.add(layers.LSTM(latent_dim, input_shape=(None, num_encoder_tokens), init='he_normal', return_sequences=False))
model.add(layers.Dropout(.5))

model.add(layers.RepeatVector(max_target_len))

for _ in range(layer_numbers):
    model.add(layers.LSTM(latent_dim, return_sequences=True, init='he_normal'))
    model.add(layers.Dropout(.5))

model.add(layers.TimeDistributed(layers.Dense(num_decoder_tokens, init='he_normal')))
model.add(layers.Activation('softmax'))

print(model.summary())

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('../data/model/best_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
callbacks_list = [checkpoint]

print('Training Model..')
model.fit(input_vector, target_vector, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=.2)