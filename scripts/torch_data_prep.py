import pickle

import torchtext
from torchtext.data.utils import get_tokenizer

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from utils import preprocess_sentence, generate_errors


max_seq_len = 200

# Read in sentences
print('Reading in Data...')
with open('data/english_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)


# preprocess texts
print('Preprocessing Data...')
preprocessed_data = []
for sent in tqdm(sentences):
    result = preprocess_sentence(sent)
    if len(result) > 0:
        preprocessed_data.append(result)

# generate spelling errors
print('Generating Spelling Errors...')
encoder_data = []
for sent in tqdm(preprocessed_data):
    encoder_data.append(generate_errors(sent))
target_data = preprocessed_data

# tokenize the input
nesting_field = torchtext.data.Field(batch_first=True)
tokenizer = torchtext.data.NestedField(
    nesting_field,
    init_token='<s>',
    eos_token='</s>',
    tokenize=list,
)

tokenizer.build_vocab(encoder_data + target_data)

print('Padding Encoder Data...')
for i in tqdm(range(len(encoder_data))):
    encoder_data[i] = list(encoder_data[i])
    if len(encoder_data[i]) > max_seq_len:
        encoder_data[i] = encoder_data[i][:max_seq_len]
    else:
        encoder_data[i] = encoder_data[i] + ['<pad>'] * (max_seq_len-len(encoder_data[i]))

print('Padding Target Data...')
for i in tqdm(range(len(target_data))):
    target_data[i] = list(target_data[i])
    if len(target_data[i]) > max_seq_len:
        target_data[i] = target_data[i][:max_seq_len]
    else:
        target_data[i] = target_data[i] + ['<pad>'] * (max_seq_len-len(target_data[i]))

print('Tokenizing Data...')
encoder_data = tokenizer.numericalize([encoder_data])
target_data = tokenizer.numericalize([target_data])

print('Saving Tokenizer, and tokenized Data...')
with open('data/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('data/torch_encode_data.pkl', 'wb') as f:
    pickle.dump(encoder_data, f, protocol=4)

with open('data/torch_decode_data.pkl', 'wb') as f:
    pickle.dump(target_data, f, protocol=4)