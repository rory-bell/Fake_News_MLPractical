import numpy as np
import pandas as pd
from numpy import random
import os

import torch
import torch.nn as nn

from tqdm import tqdm   

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset, DataLoader


class TransformerFakeNews(nn.Module):
    def __init__(self, params):
        super(TransformerFakeNews, self).__init__()
        self.transformer = nn.Transformer(
            d_model=params['input_size'],
            nhead=params['num_heads'],
            num_encoder_layers=params['num_layers'],
            num_decoder_layers=params['num_layers'],
        )
        self.fc = nn.Linear(params['input_size'], params['hidden_size']) # 1 is size after pooling
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(params['dropout'])
        self.output_layer = nn.Linear(params['hidden_size'], params['num_classes'])

    def forward(self, x):
        x = self.transformer(x, x)
        # x = x.mean(dim=1)  # Aggregate over sequence length (mean pooling)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=params['input_size'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            batch_first=True
        )
        # self.fc = nn.Linear(params['hidden_size'], params['hidden_size'])
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(params['dropout'])
        self.output_layer = nn.Linear(2*params['hidden_size'], params['num_classes'])
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x, _ = self.lstm(x)
        # x = self.fc(x[:, -1, :])
        # x = self.relu(x)
        x = self.output_layer(x[:, -1, :])
        x = self.relu(x)
        return x



class FakeNewsDataset2(Dataset):
    def __init__(self, dataframe_floats, tweet_tensor, statement_tensor, target_column):

        self.features = torch.tensor(dataframe_floats.drop(columns=[target_column]).values, dtype=torch.float32)
        self.features = torch.cat((self.features, tweet_tensor, statement_tensor), 1)
        #append 3 useless tensors to the features tensor to make dimensions match
        self.features = torch.cat((self.features, torch.zeros((len(self.features), 3))), 1)

        self.target = torch.tensor(dataframe_floats[target_column].values, dtype=torch.float32)

    # def __init__(self, features, target):
    #     self.features = torch.tensor(features)#, dtype=torch.float32)
    #     self.target = torch.tensor(target)#, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

def save_model(model, params):
        save_path = os.path.join(params['save_path'], 'model.pt')   
        op_state = {'state_dict': model.state_dict(), 'params' : params}
        #create directory if it doesn't exist
        os.path.isdir(params['save_path']) or os.makedirs(params['save_path'])
        torch.save(op_state, save_path)

def load_model_dict(model, params):
    model.load_state_dict(torch.load(params['save_path'] + 'model.pt')['state_dict'])
    return model

def train_tf(params, model, train_loader, loss, optimizer):
    
    total_step = len(train_loader)
    print(total_step)
    for epoch in range(params['num_epochs']):
        for i, (features, labels) in enumerate(tqdm(train_loader)):
            features = features.float()
            labels = labels.float()
            # print(features.shape, labels.shape)
            outputs = model(features)
            l = loss(outputs.reshape(-1,1), labels.reshape(-1,1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, params['num_epochs'], i+1, total_step, l.item()))
        save_model(model, params)