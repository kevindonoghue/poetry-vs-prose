import numpy as np
import torch
import torch.nn as nn
import json
import re
from pprint import pprint
import argparse
import os


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    device = 'cuda'
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    device = 'cpu'
    
   
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='my_model')
parser.add_argument('--num-iterations', type=int, default=100000)
parser.add_argument('--source-filename', type=str, default='../../data/poem_and_prose_data.json')
args = parser.parse_args()

os.makedirs(args.model_name, exist_ok=True)
    

# source is a json file created by clean_prose_and_poetry_data.py
with open(args.source_filename) as f:
    papd = json.load(f)
    text = dict()
    text['prose'] = papd['prose']
    text['poems'] = papd['poems']
    id2token = {int(i): token for i, token in papd['id2token'].items()}
    token2id = {token: int(i) for token, i in papd['token2id'].items()}

np.random.shuffle(text['prose'])
np.random.shuffle(text['poems'])
prose_limit = int(len(text['prose'])*0.8)
poems_limit = int(len(text['poems'])*0.8)
train_text = dict()
test_text = dict()
train_text['prose'] = text['prose'][:prose_limit]
train_text['poems'] = text['poems'][:poems_limit]
test_text['prose'] = text['prose'][prose_limit:]
test_text['poems'] = text['poems'][poems_limit:]

len_lexicon = len(id2token)
id2token[len_lexicon] = '<?>'
id2token[len_lexicon+1] = '<pad>'
token2id['<?>'] = len_lexicon
token2id['<pad>'] = len_lexicon+1
id2token[len_lexicon+2] = '<start>'
token2id['<start>'] = len_lexicon+2
id2token[len_lexicon+3] = '<end>'
token2id['<end>'] = len_lexicon+3
len_lexicon = len(id2token)

# need to preserve the randomly generated train and test sets
with open(os.path.join(args.model_name, 'train_test_data.json'), 'w+') as f:
    json.dump({'train': train_text, 'test': test_text, 'id2token': id2token, 'token2id': token2id}, f)

SEQUENCE_LENGTH = 16
BATCH_SIZE = 64
EMBEDDING_DIM = 30
HIDDEN_DIM = 25
NUM_LAYERS = 1
NUM_DIRECTIONS = 2

def unescape(s):
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&amp;', '&')
    return s
       
def convert_to_numerical(arr):
    # arr is a list of tokens to be converted to numerical values
    return_arr = []
    for token in arr:
        if token in token2id:
            return_arr.append(token2id[token])
        else:
            return_arr.append(token2id['<?>'])
    return return_arr

def get_sequence_batch(batch_size, sequence_length, phase='train'):
    if phase == 'train':
        text = train_text
    else:
        text = test_text

    token_sequences = []
    numerical_sequences = []
    targets = []
    length = np.random.randint(5, sequence_length-1)
    for _ in range(batch_size):
        form = np.random.choice(['prose', 'poems'])

        text_sample = text[form][np.random.choice(len(text[form]))]
        if form == 'poems':
            tokens = [unescape(y.lower()) for y in re.split('\W+', text_sample) if y != '']
        else:
            tokens = [y.lower() for y in re.split('\W+', text_sample) if y != '']
        while len(tokens) <= length:
            text_sample = text[form][np.random.choice(len(text[form]))]
            if form == 'poems':
                tokens = [unescape(y.lower()) for y in re.split('\W+', text_sample) if y != '']
            else:
                tokens = [y.lower() for y in re.split('\W+', text_sample) if y != '']

        subseq_start = np.random.randint(len(tokens)-length)
        subseq = tokens[subseq_start:subseq_start+length]
        subseq = convert_to_numerical(subseq)
        numerical_sequences.append(subseq)
        token_sequences.append(tokens[subseq_start:subseq_start+length])
        if form == 'poems':
            target = 1
        else:
            target = 0
        targets.append(target)
    return LongTensor(numerical_sequences), token_sequences, LongTensor(targets)
        
        
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_directions = NUM_DIRECTIONS
        self.num_layers = NUM_LAYERS
        bidirectional = self.num_directions == 2
        self.embed = nn.Embedding(len_lexicon, EMBEDDING_DIM)
        self.gru = nn.GRU(input_size=EMBEDDING_DIM,
                        hidden_size=HIDDEN_DIM,
                        num_layers=self.num_layers,
                        batch_first=True,
                        bidirectional=bidirectional)
        self.fc = nn.Linear(HIDDEN_DIM*self.num_directions*self.num_layers, 2)
        
    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        h = h.permute(1, 0, 2)
        h = h.reshape(-1, self.num_layers*self.num_directions*HIDDEN_DIM)
        out = self.fc(h)
        return out
    
    def fit(self, num_iterations, loss_fn, optimizer, print_every=100):
        for i in range(num_iterations):
            numerical_sequences, token_sequences, targets = get_sequence_batch(BATCH_SIZE, SEQUENCE_LENGTH)
            out = self.forward(numerical_sequences)
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % print_every == 0:
                print('loss: ', loss.item())
                _, token_sequences, targets = get_sequence_batch(4*BATCH_SIZE, SEQUENCE_LENGTH, 'test')
                num_correct = 0
                for j, seq in enumerate(token_sequences):
                    pred = self.predict(seq)
                    actual = targets[j].item()
                    if actual == pred:
                        num_correct += 1
                    if j < 5:
                        print(' '.join(seq))
                        print('prediction: ', 'poetry' if pred==1 else 'prose')
                        print('actual:     ', 'poetry' if actual==1 else 'prose')
                        print('')
                print('acc: ', num_correct/(4*BATCH_SIZE))
                print('')
                print('')
                    
    def predict(self, token_sequence):
        numerical_sample = convert_to_numerical(token_sequence)
        numerical_sample = LongTensor(numerical_sample).view(1, -1)
        with torch.no_grad():
            self.eval()
            out = self.forward(numerical_sample)
            pred = out.cpu().numpy().reshape(-1).argmax()
            self.train()
            return pred
        
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


model.fit(args.num_iterations, loss_fn, optimizer, print_every=100)

torch.save(model.state_dict(), os.path.join(args.model_name, 'saved_model.pt'))
