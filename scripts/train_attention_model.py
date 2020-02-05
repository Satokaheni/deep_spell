import numpy as np
import pickle
import os
import time
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split

import tensorflow as tf

from attention_model import *


print('Reading in the data...')

with open('../data/input_char_tables_v3.pkl', 'rb') as f:
    input_char2id, input_id2char = pickle.load(f)

with open('../data/target_char_tables_v3.pkl', 'rb') as f:
    target_char2id, target_id2char = pickle.load(f)

with open('../data/vectorized_input_v3.pkl', 'rb') as f:
    input_vector, _, target_vector = pickle.load(f)

print('Creating input vectors...')

encode_input_vector = np.zeros((input_vector.shape[0], input_vector[0].shape[0]))
for i in range(input_vector.shape[0]):
    encode_input_vector[i, :] = np.array([np.argmax(input_vector[i][j]) for j in range(input_vector[i].shape[0])])

target_input_vector = np.zeros((target_vector.shape[0], target_vector[0].shape[0]))
for i in range(target_vector.shape[0]):
    target_input_vector[i, :] = np.array([np.argmax(target_vector[i][j]) for j in range(target_vector[i].shape[0])])

print(encode_input_vector.shape)
print(target_input_vector.shape)

print('Splitting into train/test...')
train_input, val_input, train_target, val_target = train_test_split(encode_input_vector, target_input_vector, test_size=.2)

batch_size = 32
epochs = 200
val_epoch = 10
latent_dim = 256
embed_dim = 128
attention_dim = 20
num_encoder_tokens = len(input_char2id) + 1
num_decoder_tokens = len(target_char2id) + 1
layer_numbers = 2
max_input_len = 262
max_target_len = 260
buffer_size = len(train_input)
steps_per_epoch = buffer_size // batch_size
steps_per_epoch_val = len(val_input) // batch_size

print('Creating TF Dataset...')
dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset_val = tf.data.Dataset.from_tensor_slices((val_input, val_target)).shuffle(len(val_input))
dataset_val = dataset_val.batch(batch_size, drop_remainder=True)

print('Creating Model..')
encoder = Encoder(num_encoder_tokens, embed_dim, latent_dim, batch_size)
attention = BahdanauAttention(attention_dim)
decoder = Decoder(num_decoder_tokens, embed_dim, latent_dim, batch_size)

optimizer = tf.keras.optimizers.Adam()
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_func(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_obj(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

checkpoint_dir = '../data/model/'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([target_char2id['\t']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_func(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss/ int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

@tf.function
def val_step(inp, targ, enc_hidden):
    loss = 0

    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([target_char2id['\t']] * batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        loss += loss_func(targ[:, t], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss/ int(targ.shape[1]))

    return batch_loss


def evaluate(sentence):
    attention_plot = np.zeros((max_target_len, max_input_len))

    sentence = sentence.lower()

    inputs = [input_char2id[char] for char in list(sentence)]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_input_len, padding='post', value=input_char2id[' '])

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, latent_dim))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_char2id['\t']], 0)

    for t in range(max_target_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_id2char[predicted_id]

        if target_id2char[predicted_id] == '\n':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def spell_correct(sentence):
    result, sentence, _ = evaluate(sentence)

    return result

print('Training Model...')
for epoch in range(epochs):
    start = time.time()

    best_loss = np.inf

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch: {}\tBatch: {}\tLoss: {:.4f}'.format(epoch+1, batch, batch_loss.numpy()))

    print('Epoch: {}\tLoss: {:.4f}\tEpoch Time: {} min'.format(epoch+1, total_loss/steps_per_epoch, (time.time() - start // 60)))

    if (total_loss/steps_per_epoch) <= best_loss:
        print('Improved upon previous best loss: {}\nCheckpointing....'.format(best_loss))
        checkpoint.save(file_prefix = checkpoint_prefix)


    if (epoch + 1) % val_epoch == 0:
        # validation
        print('Validating Model')
        for (batch, (inp, targ)) in enumerate(dataset_val.take(steps_per_epoch_val)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Validation Batch: {}\tValidation Loss: {:.4f}'.format(batch, batch_loss.numpy()))

        print('Validation Loss: {:.4f}'.format(total_loss/steps_per_epoch_val))

