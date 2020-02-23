import pickle
import numpy as np
from time import time
from tqdm import tqdm
from torch_model import TransformerModel
from utils import accuracy_score

import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load in data
print('Reading in Data...')
with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('data/torch_encode_data.pkl', 'rb') as f:
    input_data = pickle.load(f)[0]

with open('data/torch_decode_data.pkl', 'rb') as f:
    target_data = pickle.load(f)[0]


ntokens = len(tokenizer.vocab.stoi)
emsize = 150
nhid = 128
nlayers = 2
nhead = 2
dropout = .5
batch_size = 32
epochs = 100
val_epoch_step = 10

print('Creating Torch Datasets')
train_input, val_input, train_targets, val_targets = train_test_split(input_data.numpy(), target_data.numpy(), test_size=.2)
train_dataset = TensorDataset(torch.tensor(train_input), torch.tensor(train_targets))
val_dataset = TensorDataset(torch.tensor(val_input), torch.tensor(val_targets))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


print('Creating Model...')
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model.cuda()

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
softmax = torch.nn.Softmax(dim=1)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

print('Training Model...')

best_loss = np.inf
for i in range(1, epochs+1):
    print('------------------------------------\nEpoch {}\n------------------------------------'.format(i))
    model.train()
    batch_loss = []
    batch_accuracy = []
    start = time()

    y_true = []
    y_preds = []

    t = tqdm(iter(train_dataloader), leave=False, total=len(train_dataloader))
    for _, batch in enumerate(t):
        x = batch[0].to(device)
        y = batch[0].to(device)

        optimizer.zero_grad()

        preds = model(x)

        loss = loss_func(preds.view(-1, ntokens), y.view(-1))
        
        preds = softmax(preds)
        preds = np.array([[np.argmax(char_pred) for char_pred in batch] for batch in preds.detach().cpu().numpy()])

        y = y.detach().cpu().numpy()
        y_preds.extend(preds)
        y_true.extend(y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        scheduler.step()

        batch_loss.append(loss.item())

    batch_loss = np.array(batch_loss).mean()
    accuracy = accuracy_score(y_true, y_preds)
    print('Training Loss: {}\tTraining Accuracy: {}\tTraining Epoch Time: {} mins'.format(batch_loss, accuracy, (time() - start) // 60))
    
    if batch_loss <= best_loss:
        print('Acheived New best Loss\nSaving Model')
        torch.save(model.state_dict(), 'data/toch_model.h5')
        best_loss = batch_loss

    if i % val_epoch_step == 0:
        print('------------------------------------\nValidating Model\n------------------------------------')

        model.eval()
        val_batch_loss = []
        y_preds = []
        y_true = []
        val_start = time()

        t = tqdm(iter(val_dataloader), leave=False, total=len(val_dataloader))
        for _, batch in enumerate(t):
            x = batch[0].to(device)
            y = batch[1].to(device)

            with torch.no_grad():
                preds = model(x)

            val_loss = loss_func(preds.view(-1, ntokens), y.view(-1))
            val_batch_loss.append(val_loss.item())

            preds = softmax(preds)
            preds = [[np.argmax(char_pred) for char_pred in batch] for batch in preds.detach().cpu().numpy()]

            y = y.detach().cpu().numpy()
            y_preds.extend(preds)
            y_true.extend(y)

        val_batch_loss = np.array(val_batch_loss).mean()
        accuracy = accuracy_score(y_true, y_preds)
        print('Validation Loss: {}\tValidation Accuracy: {}\tValidation Epoch Time: {} mins'.format(val_batch_loss, accuracy, (time() - val_start) // 60))

