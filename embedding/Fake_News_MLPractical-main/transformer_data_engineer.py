import numpy as np
import pandas as pd
from numpy import random

import torch
import torch.nn as nn
from stemming.porter2 import stem
from torch.utils.data import Dataset, DataLoader

import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA


def get_dataset(params):
    def preprocess(data, stemming=False, stopping=True):
        with open('stop_words.txt', 'r') as stop_file:
            stop_words = set(line.strip() for line in stop_file)
        tokens_array = []
        for text in data:
            text = text.replace('-', ' ').replace('/', ' ').replace('"', '').replace('\n', ' ') 
            text = re.sub(r'\s+', ' ', text) 
            text = re.sub('[^a-zA-Z" "\n]+', '', text)
            tokens = text.lower().split()
            if stopping:
                tokens = [word for word in tokens if word not in stop_words]
            if stemming:
                tokens = [stem(word) for word in tokens]
            tokens_array.append(tokens)
    
        return tokens_array    

    def generate_embeddings(tokens, filepath, embedding_dim=50):
        # Initialize the tokenizer and fit on the tweet tokens
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tokens)
        
        # Create the embedding matrix from the GloVe embeddings
        def embedding_for_vocab(filepath, word_index, embedding_dim):
            vocab_size = len(word_index) + 1
            embedding_matrix_vocab = np.zeros((vocab_size, embedding_dim))
            with open(filepath, encoding="utf8") as f:
                for line in f:
                    word, *vector = line.split()
                    if word in word_index:
                        idx = word_index[word]
                        embedding_matrix_vocab[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
            return embedding_matrix_vocab
        
        # Define the positional encoding function
        def positional_encoding(length, depth):
            position = np.arange(length)[:, np.newaxis]
            div_term = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth))
            pe = np.zeros((length, depth))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            return pe
        
        embedding_matrix_vocab = embedding_for_vocab(filepath, tokenizer.word_index, embedding_dim)
        
        # Convert tweet tokens to sequences
        sequences = tokenizer.texts_to_sequences(tokens)
        
        # Pad sequences to have the same length
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Generate embeddings with positional encodings
        token_embeddings = []
        for seq in padded_sequences:
            token_embedding = np.array([embedding_matrix_vocab[idx] for idx in seq])
            pos_encoding = positional_encoding(max_length, embedding_dim)
            token_embedding += pos_encoding
            token_embeddings.append(token_embedding)
        
        return token_embeddings


    categories = ['Services', 'Pandemic', 'Economy', 'Discord', 'Culture',
        'Elections', 'Environment', 'Industry']

    from enum import Enum, auto

    class ThreeLabelMajority(Enum):
        MostlyAgree = 1
        Agree = 2
        NOMAJORITY = 3
        MostlyDisagree = 4
        Disagree = 5

    #convert data['threeLabelMajority'] to enum (first remove spaces)


    data = pd.read_csv('../../data/CategorisedFakeNewsTweetsFinal.csv')
    data = data.drop(['majorityTarget', 'fiveLabelMajority'], axis=1)
    data['threeLabelMajority'] = data['threeLabelMajority'].str.replace(' ', '').apply(lambda x: ThreeLabelMajority[x])
    data['threeLabelMajority'] = data['threeLabelMajority'].apply(lambda x: float(x.value))
    statements = data['statement']
    tweets = data['tweet']

    statement_tokens = preprocess(statements)
    tweet_tokens = preprocess(tweets)

    statement_embeddings = generate_embeddings(statement_tokens, 'glove/glove.6B.50d.txt', 50)
    tweet_embeddings = generate_embeddings(tweet_tokens, 'glove/glove.6B.50d.txt', 50)

    assert len(statement_embeddings) == len(data['statement']) and len(tweet_embeddings) == len(data['tweet']), "Length of embeddings does not match DataFrame."

    data['statement'] = statement_embeddings
    data['tweet'] = tweet_embeddings


    grouped = data.groupby('primaryCat')                      
    datasets = {}
    # Iterate over each group and create a separate dataset
    for name, group in grouped:
        datasets[name] = group







    development_set = datasets['Services'].drop(['primaryCat','secondaryCat'], axis=1)#.values
    statement_column = development_set['statement']
    tweet_column = development_set['tweet']
    development_set.drop(['statement', 'tweet'], axis=1, inplace=True)

    def squeeze_data(input):
        return torch.tensor(input).view(-1, input.shape[0]*input.shape[1]).squeeze()

    def reshape_data(input):
        return input.reshape(input.shape[0]*input.shape[1])

    statement_column = statement_column.apply(lambda x: reshape_data(x))
    # statement_column = statement_column.apply(lambda x: squeeze_data(x))
    tweet_column = tweet_column.apply(lambda x: reshape_data(x))
    # tweet_column = tweet_column.apply(lambda x: squeeze_data(x))

    alltweets = np.zeros((tweet_column.values.shape[0], tweet_column.values[0].shape[0]), dtype=np.float64)
    allstatements = np.zeros((statement_column.values.shape[0], statement_column.values[0].shape[0]), dtype=np.float64)
    for i, tweet in enumerate(tweet_column.values):
        alltweets[i] = tweet
        allstatements[i] = statement_column.values[i]

    tweet_tensor = torch.tensor(alltweets, dtype=torch.float64)
    statement_tensor = torch.tensor(allstatements, dtype=torch.float64)
    # tweet_tensor = torch.tensor(np.zeros(1))
    # statement_tensor = torch.tensor(np.zeros(1))



    import transformer_model
    import importlib
    importlib.reload(transformer_model)
    from transformer_model import FakeNewsDataset2, TransformerFakeNews

    dataset = FakeNewsDataset2(development_set, tweet_tensor, statement_tensor, 'binaryNumTarget')

    return dataset