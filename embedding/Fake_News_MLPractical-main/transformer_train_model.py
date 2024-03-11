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

import transformer_data_engineer
import transformer_model
from transformer_model import train_tf
from torch.utils.data import DataLoader


params = {}
params['input_size'] = 5500
params['hidden_size'] = 64
params['num_classes'] = 1
params['num_heads'] = 4
params['num_layers'] = 1 
assert params['input_size'] % params['num_heads'] == 0, "Input size must be divisible by the number of heads"
params['dropout'] = 0.1
params['batch_size'] = 128
params['num_epochs'] = 3
params['learning_rate'] = 0.001
params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params['save_path'] = "C:\\Users\\roryb\\OneDrive\\Documents\\Uni Work\\MLP\\Group_Project\\Fake_News_MLPractical\\embedding\\Fake_News_MLPractical-main\\models\\transformer"



dataset = transformer_data_engineer.get_dataset(params)
model = transformer_model.LSTM(params) #TransformerFakeNews(params)

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])


train_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
train_tf(params, model, train_loader, loss, optimizer)
model = transformer_model.load_model_dict(model, params)
