from keras.models import Model
from keras import layers


# function to create model
def build_model(max_input_len, input_vocab_size, max_output_len, output_vocab_len):
    
    encoder_input = layers.Input((max_input_len,))
    decoder_input = layers.Input((max_output_len,))

    encoder = layers.Embedding(input_vocab_size, 128, input_length=max_input_len, mask_zero=True)(encoder_input)

    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True, return_state=True, unroll=True), merge_mode='concat')(encoder)
    encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
    encoder_h = layers.concatenate([forward_h, backward_h])
    encoder_c = layers.concatenate([forward_c, backward_c])

    decoder = layers.Embedding(output_vocab_len, 2*128, input_length=max_output_len, mask_zero=True)(decoder_input)

    decoder = layers.LSTM(2*128, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_h, encoder_c])

    attention = layers.dot([decoder, encoder_outs], axes=[2,2])
    attention = layers.Activation('softmax', name='attention')(attention)

    context = layers.dot([attention, encoder_outs], axes=[2,1])

    decoder_combined_context = layers.concatenate([context, decoder])

    decoder_combined_dropout = layers.Dropout(.5)(decoder_combined_context)

    output = layers.TimeDistributed(layers.Dense(128, activation='tanh'))(decoder_combined_dropout)
    output = layers.TimeDistributed(layers.Dense(output_vocab_len, activation='softmax'))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model