## Overview
An LSTM-based poetry vs prose classifier trained on 19th century works scraped from Project Gutenberg and poetryfoundation.org. It performs with ~80% accuracy. Two human experts tested on the same dataset had an accuracy of 87%.

## Model Details
Works of prose and poetry are collected, removed of all punctuation and line breaks, rendered lowercase, tokenzied, then divided into sequences of tokens of lengths 5 to 16. These sequences are then encoded into the hidden layer of an LSTM which is trained to classify them as poetry or prose. Therefore, both order and word count are taken into consideration. Perhaps surprisingly, this model performs the same as a simpler bag of words model where the sequences are encoded into tfidf vectors and classified via logistic regression.