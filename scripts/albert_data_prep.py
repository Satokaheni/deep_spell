import pickle
from utils import preprocess_sentence
from tqdm import tqdm

# Read in sentences
print('Reading in Data...')
with open('data/english_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

print('Generating Input/Output Pairs...')
input_sentences = []
target_sentences = []
for sent in tqdm(sentences):
    # preprocess sentence
    sent = preprocess_sentence(sent)
    if len(sent.split()) > 1:
        target_sentences.append(sent)

        # choose a random word to mask
        sent = sent.split()
        mask_word = np.random.choice(list(range(len(sent))))
        sent[mask_word] = '[MASK]'
        input_sentences.append(' '.join(sent))


with open('data/albert_input.pkl', 'wb') as f:
    pickle.dump(input_sentences, f, protocol=4)

with open('data/albert_target.pkl', 'wb') as f:
    pickle.dump(target_sentences, f, protocol=4)