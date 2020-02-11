import numpy as np

import spacy

nlp = spacy.load('en_core_web_sm')
CHARS = 'abcdefghijklmnopqrstuvwxyz'
START_CHAR = 1
PAD_CHAR = 0


# Create Encoding and Decoding Dictionary for inputs and ouputs
# Use 2 as start since 1 will be the start of sentence token and 0 is padding
def create_encode_decode_tables(input_data):

    encoding_dict = {}
    decoding_dict = {}

    for sent in input_data:
        for char in sent:
            if char not in encoding_dict:
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(encoding_dict)] = char

    return encoding_dict, decoding_dict, len(encoding_dict) + 2


# Encode sentences into character integer sequences
def encode_sentences(encode_dict, sentences, max_sent_length):
    
    encoded_sentences = np.zeros((len(sentences), max_sent_length))
    for i in range(len(sentences)):
        for j in range(min(len(sentences[i]), max_sent_length)):
            encoded_sentences[i, j] = encode_dict[sentences[i][j]]

    return encoded_sentences


# Decode the output of the model
def decode_sentence(decode_dict, sentence):

    return ''.join([decode_dict[x] for x in sentence if x != 0])


# function to preprocess a sentence before encoding
def preprocess_sentence(sentence):

    # We are going to replace all proper nouns with special tokens
    # also we are going to lowercase
    sent = nlp(sentence)
    return ' '.join([w.text.lower() if w.tag_ != 'NNP' else '<nnp>' for w in sent])


# function to generate spelling errors
def generate_errors(sentence):
    '''
        first decide how many imputations to make then randomly decide between 
            1. removing a space 
            2. removing a character in a word 
            3. swapping two characters in a word 
            4. adding a character to a word
    '''

    # determine number of imputations we'll max out at 4
    number_imps = np.random.randint(1, 4, size=1)

    # determine which errors to create
    errors = np.random.choice([1, 2, 3, 4], size=number_imps)

    # split the sentence by spaces to make it easier for choices
    sent_split = sentence.split()

    # randomly choose a words for the errors that will happen
    # allow replacement multiple mispellings for the same word
    non_space_errors = [e for e in errors if e != 1]
    
    # remove <nnp> special tokens as choices
    allowed_words = []
    for i in range(len(sent_split)):
        if sent_split[i] != '<nnp>':
            allowed_words.append(i)
    
    word_choices = np.random.choice(allowed_words, size=len(non_space_errors))

    # iterate through words being misspelled
    for word in word_choices:

        # randomly choose an option of remove, swap or add
        option = np.random.randint(1, 3)
        
        # randomly remove a character
        if option == 1:
            char_choice = np.random.randint(0, len(sent_split[word]))
            sent_split[word] = sent_split[word][:char_choice] + sent_split[word][char_choice+1:]

        # randomly swap two characters 
        elif option == 2:
            char_choice = np.random.randint(0, len(sent_split[word])-1)
            sent_split[word] = sent_split[word][:char_choice] + sent_split[word][char_choice+1] + sent_split[word][char_choice] + sent_split[char_choice+2:]

        # randomly add a character
        else:
            char_choice = np.random.randint(0, len(sent_split[word]))
            added_char = int(np.random.randint(0, 26, size=1))
            added_char = CHARS[added_char]
            sent_split[word] = sent_split[word][:char_choice] + added_char + sent_split[char_choice:]

    # remove spaces last
    # remove a space/spaces
    space_removals = [e for e in errors if e == 1]
    for _ in space_removals:

        # randomly choose a space to remove
        remove = np.random.randint(0, len(sent_split)-1)
        sentence = ' '.join(sent_split[:remove+1]) + ' '.join(sent_split[remove+1:])
        sent_split = sentence.split()

    return ' '.join(sent_split)


# Create training data
def create_training_data(input_data):
    
    # preprocess data
    preproccesed_data = [preprocess_sentence(x) for x in input_data]

    # create spelling errors
    encoder_data = [generate_errors(x) for x in preproccesed_data]
    target_data = preproccesed_data

    # encoder input
    encoding_w2id, encoding_id2w, encoding_dict_len = create_encode_decode_tables(encoder_data)
    encoder_data = encode_sentences(encoding_w2id, encoder_data, max_sent_length=200)

    # target output
    decoder_w2id, decoder_id2w, decoder_dict_len = create_encode_decode_tables(target_data)
    target_data = encode_sentences(decoder_w2id, target_data, max_sent_length=200)

    # decoder 
    decoder_input = np.zeros_like(target_data)
    decoder_input[:, 1:] = encoder_data[:, :-1]
    decoder_input[:, 0] = START_CHAR
    decoder_output = np.eye(decoder_dict_len)[target_data.astype('int')]

    x = [encoder_data, decoder_input]
    y = [decoder_output]

    return x, y, {'id2w': encoding_id2w, 'w2id': encoding_w2id}, {'id2w': decoder_id2w, 'w2id': decoder_w2id}


# function to generate encoded outputs for inputs for testing
def generate(texts, encoding_w2id, model, max_input_len, max_output_len, beam_size=3, max_beams=3, min_cut_off_len=10, cut_off_ratio=1.5):

    # choose one
    # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
    min_cut_off_len = min(min_cut_off_len, max_output_len)

    all_completed_beams = {i: [] for i in range(len(texts))}
    all_running_beams = {}

    for i, text in enumerate(texts):
        all_running_beams[i] = [[np.zeros((len(text), max_output_len)), [1]]]
        all_running_beams[i][0][0][:, 0] = START_CHAR

    while len(all_running_beams) != 0:
        for i in all_running_beams:
            all_running_beams[i] = sorted(all_running_beams[i], key=(lambda tup: np.prod(tup[1])), reverse=True)
            all_running_beams[i] = all_running_beams[i][:max_beams]

        in_out_map = {}
        batch_encoder_input = []
        batch_decoder_input = []

        t_c = 0 
        for text_i in all_running_beams:
            if text_i not in in_out_map:
                in_out_map[text_i] = []
            for running_beam in all_running_beams[text_i]:
                in_out_map[text_i].append(t_c)
                t_c += 1
                batch_encoder_input.append(texts[text_i])
                batch_decoder_input.append(running_beam[0][0])

        batch_encoder_input = encode_sentences(encoding_w2id, batch_encoder_input, max_input_len)
        batch_decoder_input = np.asarray(batch_decoder_input)
        batch_predictions = model.predict([batch_encoder_input, batch_decoder_input])

        t_c = 0
        for text_i, _ in in_out_map.items():
            temp_running_beams = []
            for running_beam, probs in all_running_beams[text_i]:
                if len(probs) >= min_cut_off_len:
                    all_completed_beams[text_i].append([running_beam[:, 1:], probs])
                else:
                    prediction = batch_predictions[t_c]
                    sorted_args = prediction.argsort()
                    sorted_probs = np.sort(prediction)

                    for i in range(1, beam_size+1):
                        temp_running_beam = np.copy(running_beam)
                        i = -1 * i
                        ith_arg = sorted_args[:, i][len(probs)]
                        ith_prob = sorted_probs[:, i][len(probs)]

                        temp_running_beam[:, len(probs)] = ith_arg
                        temp_running_beams.append([temp_running_beam, probs + [ith_prob]])
                    
                t_c += 1

            all_running_beams[text_i] = [b for b in temp_running_beams]

        to_del = []
        for i, v in all_running_beams.items():
            if not v:
                to_del.append(i)

        for i in to_del:
            del all_running_beams[i]

    return all_completed_beams


# function to generate outputs for given inputs
def infer(texts, model, encoding_w2id, decoding_id2w, max_input_len, max_output_len):

    all_decoder_outputs = generate(texts, encoding_w2id, model, max_input_len, max_output_len)
    outputs = []

    for _, decoder_outputs in all_decoder_outputs.items():
        outputs.append([])
        for decoder_output, probs in decoder_outputs:
            outputs[-1].append({'sequence': decode_sentence(decoding_id2w, decoder_output[0]), 'prob': np.prod(probs)})

    return outputs