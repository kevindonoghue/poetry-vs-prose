import json
from gensim import corpora
import re
import os
from pprint import pprint
from collections import defaultdict



# handle prose data
authors = \
"""
Jane Austen
Charles Dickens
Bronson Alcott
Arthur Conan Doyle
Henry James
Rudyard Kipling
H.G. Wells
L. Frank Baum
Nathaniel Hawthorne
Ralph Waldo Emerson
Edgar Allen Poe
Mark Twain
Ralph Waldo Emerson
"""

authors = authors.split('\n')[1:-1]


def get_body(s):
    body_pattern = r'(?<=\*\*\*\n).+(?=End of the Project Gutenberg)'
    search = re.search(body_pattern, s, flags=re.DOTALL)
    if search:
        return search.group(0).strip('\n')

def get_title(s):
    title_pattern = r'(?<=Title\: ).+'
    search = re.search(title_pattern, s)
    if search:
        return search.group(0)

def get_author(s):
    author_pattern = r'(?<=Author\: ).+'
    search = re.search(author_pattern, s)
    if search:
        return search.group(0)

def get_language(s):
    language_pattern = r'(?<=Language\: ).+'
    search = re.search(language_pattern, s)
    if search:
        return search.group(0)
    
def body_to_tokens(body):
    return [y.lower() for y in re.split('\W+', body) if y != '']
    
bodies = []

for directory_name in os.listdir('../data/arranged_gutenberg_documents'):
    if any([author in directory_name for author in authors]):
        directory = os.path.join('../data/arranged_gutenberg_documents', directory_name)
        for filename in os.listdir(directory):
            if 'poet' not in filename and 'poem' not in filename:
                with open(os.path.join(directory, filename)) as f:
                    s = f.read()
                body = get_body(s)
                language = get_language(s)
                if body and language == 'English':
                    document_bodies = [x for x in body.split('\n\n') if len(x) > 200]
                    bodies.extend(document_bodies[3:-3])
                

token_sets = [body_to_tokens(body) for body in bodies]
vocab_counts = defaultdict(int)
for token_set in token_sets:
    for token in token_set:
        vocab_counts[token] += 1

ordered_vocab = sorted(vocab_counts.items(), key=lambda t: -t[1])
ordered_vocab = [x[0] for x in ordered_vocab]

limit = int(0.2*len(ordered_vocab))
id2token = {i: token for i, token in enumerate(ordered_vocab[:limit])}
token2id = {token: i for i, token in enumerate(ordered_vocab[:limit])}


prose_token_data = {'bodies': bodies,
                   'token2id': token2id,
                   'id2token': id2token}
####





# handle poem data
with open('../data/poems.json') as f:
    poems = json.load(f)
  
  
def unescape(s):
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&amp;', '&')
    return s
  
bodies = [unescape(entry['body']) for entry in poems]

token_sets = [body_to_tokens(body) for body in bodies]
vocab_counts = defaultdict(int)
for token_set in token_sets:
    for token in token_set:
        vocab_counts[token] += 1

ordered_vocab = sorted(vocab_counts.items(), key=lambda t: -t[1])
ordered_vocab = [x[0] for x in ordered_vocab]

limit = int(0.2*len(ordered_vocab))
id2token = {i: token for i, token in enumerate(ordered_vocab[:limit])}
token2id = {token: i for i, token in enumerate(ordered_vocab[:limit])}


poem_token_data = {'bodies': bodies,
                   'token2id': token2id,
                   'id2token': id2token}
####




# collect prose and poetry data together
all_tokens = list(set(list(poem_token_data['token2id']) + list(prose_token_data['token2id'])))
id2token = {i: token for i, token in enumerate(all_tokens)}
token2id = {token: i for i, token in enumerate(all_tokens)}

d = {'prose': prose_token_data['bodies'],
     'poems': poem_token_data['bodies'],
     'id2token': id2token,
     'token2id': token2id}

with open('../data/poem_and_prose_data.json', 'w+') as f:
    json.dump(d, f)
####
