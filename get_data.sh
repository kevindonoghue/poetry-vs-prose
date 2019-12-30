#! /bin/bash

mkdir -p data

cd scripts

# download ~50gb of data from project gutenberg
bash get_gutenberg.sh

# arrange the data by author
python3 arrange_gutenberg.py

# download ~1000 19th century poems from poetryfoundation
python3 scrape_poetry_foundation.py

# clean up the data and package a subset of it into a json file for model training
python3 clean_prose_and_poetry_data.py