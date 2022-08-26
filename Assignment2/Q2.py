import numpy as np
import pandas as pd
import sys
import time
import pickle
from io import BytesIO
import re
import math
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout,Softmax, ReLU, Module, CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils import shuffle
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert',output_hidden_states=True)

device = torch.device('cuda')

dat_train = pd.read_csv("/home/moksh/Documents/IR-A2/Hindi/train_file.csv")
dat_test = pd.read_csv("/home/moksh/Documents/IR-A2/Hindi/test_file.csv")
train_tensor = np.load('/home/moksh/Documents/IR-A2/Hindi/train_embedding.npy', allow_pickle=True)
test_tensor = np.load('/home/moksh/Documents/IR-A2/Hindi/test_embedding.npy', allow_pickle=True)
arr_train = []
arr_test = []

for i in range(train_tensor.shape[0]):
  arr_train.append(train_tensor[i].numpy())
arr_train = np.array(arr_train)

#embeddings for test set
for i in range(len(test_tensor)):
  arr_test.append(test_tensor[i].numpy())
arr_test = np.array(arr_test)


ner_label = {'B-CORP': 0,
 'B-CW': 1,
 'B-GRP': 2,
 'B-LOC': 3,
 'B-PER': 4,
 'B-PROD': 5,
 'I-CORP': 6,
 'I-CW': 7,
 'I-GRP': 8,
 'I-LOC': 9,
 'I-PER': 10,
 'I-PROD': 11}

dat_train['label'] = dat_train['token'].apply(lambda x: ner_label[x])
dat_test['label'] = dat_test['token'].apply(lambda x: ner_label[x])

##NN Forward Pass preparation + Train and Test data preparation
class trainData():

    def __init__(self,df):
        embeddings = arr_train
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        pickle.dump(scaler,open('NER_scaler_fit','wb'))
        labels = np.array(df['label'])
        self.x = torch.from_numpy(scaled_embeddings)

        self.y = torch.from_numpy(labels)
        self.n_samples = len(labels)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class testData():

    def __init__(self,df):
        embeddings = arr_test
        scaler = pickle.load(open('/content/NER_scaler_fit','rb'))
        scaled_embeddings = scaler.transform(embeddings)
        pickle.dump(scaler,open('NER_scaler_fit','wb'))
        labels = np.array(df['label'])
        self.x = torch.from_numpy(scaled_embeddings)

        self.y = torch.from_numpy(labels)
        self.n_samples = len(labels)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

#Defines the NN architecture
class nnArch(nn.Module):
    def __init__(self,n_input,df_length):
        super().__init__()
        self.hidden1 = nn.Linear(n_input,1024)
        self.hidden2 = nn.Linear(1024,512)
        self.hidden3 = nn.Linear(512,256)
        self.hidden4 = nn.Linear(256,df_length)
        self.dropout = nn.Dropout(0.25)

    def forward(self, X):
        X = F.relu(self.hidden1(X))
        X = F.relu(self.hidden2(X))
        X = F.relu(self.hidden3(X))
        X = self.hidden4(X)
        X = F.log_softmax(X,dim=1)
        return X

trainNN = trainData(dat_train)
trainDataLoad = DataLoader(trainNN, batch_size=256,shuffle=True)
n_epochs = 500
input = 768
df_length = len(set(list(dat_train['token'])))
model = nnArch(input,df_length)
model = model.to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr=0.001)
model.train()

for epoch in range(n_epochs):
    start = time.time()
    model.train()
    batch_loss = 0
    for i, (input, target) in enumerate(trainDataLoad):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        ypred = model(input)
        loss = criterion(ypred,target)
        loss.backward()
        optimizer.step()
        batch_loss += loss
    if epoch%10==0:
      print("Epoch: {}, Loss: {}".format(epoch,batch_loss/len(trainDataLoad)))

torch.save(model.state_dict(),'NER' + '_pytorch_model.pt')

def evaluateNN(testDataLoad, model):
  predictions, actuals = list(), list()
  model.eval()
  for i, (input, target) in enumerate(testDataLoad):
      input, target = input.to(device), target.to(device)
      with torch.no_grad():
          ypred = model(input)
      ypred = ypred.cpu().detach().numpy()
      actual = target.cpu().numpy()
      ypred = np.argmax(ypred, axis=1)
      actual = actual.reshape((len(actual), 1))
      ypred = ypred.reshape((len(ypred), 1))
      predictions.append(ypred)
      actuals.append(actual)
  predictions, actuals = np.vstack(predictions), np.vstack(actuals)
  F1 = f1_score(actuals, predictions, average='micro')
  return F1

testNN = testData(dat_test)
testDataLoad = DataLoader(testNN, batch_size=64,shuffle=True)

f1_train = evaluateNN(trainDataLoad,model)
f1_test = evaluateNN(testDataLoad,model)

print("F1 Score Train Set: {}".format(f1_train))
print("F1 Score Test Set: {}".format(f1_test))

