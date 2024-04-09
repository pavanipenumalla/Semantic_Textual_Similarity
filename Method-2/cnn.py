import numpy as np
import pandas as pd
import torch
from cnn_utils import preprocess, Model, train_model

# train_data = pd.read_csv('../data/stsbenchmark/train.csv')

# sentence1, sentence2, labels = preprocess(train_data)
# X1 = torch.tensor(sentence1)
# X2 = torch.tensor(sentence2)
# y = torch.tensor(labels)

# # store X1, X2, y in a file
# torch.save(X1, 'X1.pt')
# torch.save(X2, 'X2.pt')
# torch.save(y, 'y.pt')

# load X1, X2, y from a file
X1 = torch.load('X1.pt')
X2 = torch.load('X2.pt')
y = torch.load('y.pt')

# convert tensors to float32
X1 = X1.float()
X2 = X2.float()
y = y.float()

model = Model()
train_model(model, X1, X2, y, epochs=10, batch_size=32, lr=0.001)