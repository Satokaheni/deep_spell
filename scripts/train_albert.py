import torch
import pickle
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW, AlbertTokenizer, AlbertForMaskedLM
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from utils import preprocess_sentence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


print('Reading in Data...')
with open('data/albert_input.pkl', 'rb') as f:
    input_sentences = pickle.load(f)

with open('data/albert_target.pkl', 'rb') as f:
    target_sentences = pickle.load(f)

print('Tokenizing Sentences...')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
tokenized_input_ids = []
tokenized_input_masks = []
tokenized_target_ids = []

for i in tqdm(range(len(input_sentences)//10)):
    token_input = tokenizer.encode_plus(input_sentences[i], add_special_tokens=True, max_length=200, pad_to_max_length=True, return_token_type_ids=False)
    token_target = tokenizer.encode_plus(target_sentences[i], add_special_tokens=True, max_length=200, pad_to_max_length=True, return_token_type_ids=False, return_attention_mask=False)
    tokenized_input_ids.append(token_input['input_ids'])
    tokenized_input_masks.append(token_input['attention_mask'])
    tokenized_target_ids.append(token_target['input_ids'])

print('Train/Test Split...')
train_input_ids, val_input_ids, train_input_masks, val_input_masks, train_targets, val_targets = train_test_split(tokenized_input_ids, tokenized_input_masks, tokenized_target_ids, test_size=.2, random_state=42)

print('Creating Torch DataLoaders...')
train_data = TensorDataset(torch.tensor(train_input_ids), torch.tensor(train_input_masks), torch.tensor(train_targets))
val_data = TensorDataset(torch.tensor(val_input_ids), torch.tensor(val_input_masks), torch.tensor(val_targets))
train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=10, shuffle=True)

print('Creating Model...')
model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
model.cuda()

optimizer = AdamW(model.parameters(), lr=3e-8, eps=1e-8)
epochs = 100
val_epoch = 10
softmax = torch.nn.Softmax(dim=1)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

print('Training Model...')

best_loss = np.inf
for i in range(1, epochs+1):
    print('------------------------------------\nEpoch {}\n------------------------------------'.format(i))
    model.train()
    train_loss = []

    y_preds = []
    y_true = []

    t = tqdm(iter(train_dataloader), leave=False, total=len(train_dataloader))
    for _, batch in enumerate(t):
        batch = (t.to('cuda') for t in batch)
        x_ids, x_masks, y = batch

        optimizer.zero_grad()

        loss, preds = model(x_ids, attention_mask=x_masks, masked_lm_labels=y)

        preds = softmax(preds).detach().cpu().numpy()
        preds = torch.tensor([[np.argmax(word_pred) for word_pred in batch_result] for batch_result in preds]).view(-1).numpy()
        y = y.detach().cpu().view(-1).numpy()

        y_preds.extend(preds)
        y_true.extend(y)

        loss.backward()
        train_loss.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()

    train_loss = np.array(train_loss).mean()
    accuracy = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, average='weighted')

    print('Training Loss: {}\nTraining Accuracy: {}\nTraining F1: {}'.format(train_loss, accuracy, f1))

    if train_loss <= best_loss:
        print('Acheived new best loss of {} over previous best {}\nSaving best model'.format(train_loss, best_loss))
        best_loss = train_loss
        model.save_pretrained('data/spell_checker')

    if i % val_epoch == 0:
        print('------------------------------------\nValidating Model\n------------------------------------')
        model.eval()
        val_loss = []
        y_preds = []
        y_true = []

        t = tqdm(iter(val_dataloader), leave=False, total=len(val_dataloader))
        for _, batch in enumerate(t):
            batch = (t.to('cuda') for t in batch)
            x_ids, x_masks, y = batch

            with torch.no_grad():
                loss, preds = model(x_ids, attention_mask=x_masks, masked_lm_labels=y)

            val_loss.append(loss.item())

            preds = softmax(preds).detach().cpu().numpy()
            preds = torch.tensor([[np.argmax(word_pred) for word_pred in batch_result] for batch_result in preds]).view(-1).numpy()
            y = y.detach().cpu().view(-1).numpy()

            y_preds.extend(preds)
            y_true.extend(y)

        val_loss = np.array(val_loss).mean()
        accuracy = accuracy_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds, average='weighted')
        print('Validation Loss: {}\nValidation Accuracy: {}\nValidation F1: {}'.format(val_loss, accuracy, f1))