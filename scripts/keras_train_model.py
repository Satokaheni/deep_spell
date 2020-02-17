from subprocess import call

import pickle

with open('data/encode_data.pkl', 'rb') as f:
    len_data = len(pickle.load(f))

sample_size = 30000
num_batches = len_data // sample_size

print('Number of Batches: {}'.format(num_batches))

for i in range(num_batches):
    print('Training Model on Batch {}/{}'.format(i, num_batches))
    call(['python', 'scripts/train_model_batch.py', str(sample_size), str(i)])