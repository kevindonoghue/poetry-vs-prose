import requests
from pprint import pprint
import json
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_page(n):
    page_url = f'https://www.poetryfoundation.org/ajax/poems?page={n}&sort_by=publication_date&school-period=1781-1900'
    arr = requests.get(page_url).json()['Entries']
    poems = []
    for entry in arr:
        poem_info = dict()
        if 'author' in entry:
            poem_info['author'] = entry['author']
        if 'id' in entry:
            poem_info['id'] = entry['id']
        if 'link' in entry:
            poem_info['link'] = entry['link']
        if 'title' in entry:
            poem_info['title'] = entry['title']
        if 'link' in entry:
            poem_page_text = requests.get(poem_info['link']).text
            soup = BeautifulSoup(poem_page_text, 'lxml')
            json_content = soup.find('script', attrs={'type': 'application/ld+json'}).string
            d = json.loads(json_content)
            if '@graph' in d and d['@graph']:
                if 'text' in d['@graph'][0]:
                    poem_info['body'] = d['@graph'][0]['text']
                if 'inLanguage' in d['@graph'][0]:
                    poem_info['language'] = d['@graph'][0]['inLanguage']
        poems.append(poem_info)
    return poems

# pprint(get_page(1))

poems = []

for n in tqdm(list(range(1, 61))):
    poems.extend(get_page(n))
    
with open('../data/poems.json', 'w+') as f:
    json.dump(poems, f)