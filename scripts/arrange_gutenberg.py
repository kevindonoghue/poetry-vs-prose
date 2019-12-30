import re
import os
import shutil


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


os.makedirs('../data/arranged_gutenberg_documents', exist_ok=True)
base_dir = '../data/arranged_gutenberg_documents'

def arrange_gutenberg(path):
    print('handling directory ', path)
    filenames = sorted(os.listdir(path))
    for name in filenames:
        try:
            if name[-4:] == '.txt':
                with open(os.path.join(path, name)) as f:
                    s = f.read()
                author = get_author(s)
                title = get_title(s)
                if author and title:
                    os.makedirs(os.path.join(base_dir, author), exist_ok=True)
                    shutil.copy(os.path.join(path, name), os.path.join(base_dir, author, (title + '.txt').replace('/', '-')[:100]))
                print('successfully copied ', os.path.join(path, name))
        except UnicodeDecodeError: # these errors are expected and are harmless
            print('UnicodeDecodeError at ', os.path.join(path, name))
        if os.path.isdir(os.path.join(path, name)) and name.isnumeric():
            arrange_gutenberg(os.path.join(path, name))
            
arrange_gutenberg('../data/gutenberg_documents')