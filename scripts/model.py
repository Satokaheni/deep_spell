from keras.models import Sequential
from keras import layers

config = {
    'input_layers': 2,
    'output_layers': 2,
    'dropout': .5,
    'initialization': 'he_normal',
    'hidden_layers': 700,
}

def generate_model(chars, max_output_length, config=config):
    print('Buildling Model...')
    
    model = Sequential()

    # Encode the sequence using a LSTM 
    for layer_number in config['input_layers']:
        model.add(layers.LSTM(config['hidden_layers'], input_shape=(None, chars), init=config['initialization'], return_sequences=layer_number + 1 < config['input_layers']))
        model.add(layers.Dropout(config['dropout']))

    # For decoder we repeat the encoded sequence for each time step
    model.add(layers.RepeatVector(max_output_length))

    # Decode the step of the output sequence
    for layer_number in config['output_layers']:
        model.add(layers.LSTM(config['hidden_layers'], return_sequences=True, init=config['initialization']))
        model.add(config['dropout'])

    # For each step of the output sequence, decide the character it should be
    model.add(layers.TimeDistributed(layers.Dense(chars), init=config['initialization']))
    model.add(layers.Activation('softmax'))

    return model