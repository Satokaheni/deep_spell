import pickle
import numpy as np
import torch
from torch_model import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load in data
print('Reading in Data...')
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# with open('data/torch_encode_data.pkl', 'rb') as f:
#     input_data = pickle.load(f)[0][0].unsqueeze(0)

# with open('data/torch_decode_data.pkl', 'rb') as f:
#     target_data = pickle.load(f)[0][0]

ntokens = len(tokenizer.vocab.stoi)
emsize = 150
nhid = 128
nlayers = 2
nhead = 2
dropout = .5
batch_size = 32
epochs = 100
val_epoch_step = 10

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model.load_state_dict(torch.load('data/torch_model.h5'))

model.eval()
while True:
    test_sentence = input('Sentence to test: ')
    test_sentence_target = input('Correctly spelled sentence: ')
    test_sentence_encode = torch.tensor([tokenizer.vocab.stoi[char] for char in list(test_sentence)]).unsqueeze(0)
    outputs = model(test_sentence_encode)
    softmax = torch.nn.Softmax(dim=1)
    outputs = [[np.argmax(char_pred) for char_pred in batch] for batch in softmax(outputs).detach().cpu().numpy()]
    print('Input Data')
    print(test_sentence)
    print('\n----------------------------------------------------------------\n')
    print('Softmax argmax')
    print(''.join([tokenizer.vocab.itos[int(char)] for char in outputs[0]]))
    print('\n----------------------------------------------------------------\n')
    print('Actual Sentence')
    print(test_sentence_target)