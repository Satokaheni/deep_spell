import numpy as np

import spacy

nlp = spacy.load('en_core_web_sm')
CHARS = 'abcdefghijklmnopqrstuvwxyz'


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
    return ' '.join([w.text.lower() if w.tag_ != 'NNP' else '<NNP>' for w in sent])


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

    # remove a space
    # randomly choose a space to remove
    sent_split = sentence.split()
    remove = np.random.randint(0, len(sent_split)-1)

    sentence = ' '.join(sent_split[:remove+1]) + ' '.join(sent_split[remove+1:])
    