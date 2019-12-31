import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflowjs as tfjs

import json
import os
import argparse
import re
import os


# source is a json file created by clean_prose_and_poetry_data.py
with open('../../data/poem_and_prose_data.json') as f:
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
token2id['<?>'] = len_lexicon
id2token[len_lexicon+1] = '<pad>'
token2id['<pad>'] = len_lexicon+1
len_lexicon = len(id2token)

# need to preserve the randomly generated train and test sets
with open('train_test_data.json', 'w+') as f:
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

def get_sequence_data(sample_size, sequence_length, phase='train'):
    if phase == 'train':
        text = train_text
    else:
        text = test_text

    X = []
    y = []

    for _ in range(sample_size):
        length = np.random.randint(5, sequence_length-1)
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
        subseq += ['<pad>']*(sequence_length-length)
        subseq = convert_to_numerical(subseq)
        X.append(subseq)
        if form == 'poems':
            target = 1
        else:
            target = 0
        y.append(target)
    return np.array(X), np.array(y)

X_train, y_train = get_sequence_data(100000, 16, phase='train')
X_test, y_test = get_sequence_data(10000, 16, phase='test')


model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=len_lexicon, output_dim=HIDDEN_DIM))
model.add(layers.GRU(HIDDEN_DIM, return_sequences=False))
model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE,
          epochs=1)

os.makedirs('saved_model', exist_ok=True)
tfjs.converters.save_keras_model(model, 'saved_model')
