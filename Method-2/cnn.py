import numpy as np
import pandas as pd
import torch
from cnn_utils import preprocess, Model, train_model
import torch.utils.data as data_utils

# train_data = pd.read_csv('../data/sick/train.csv')

# train_data = pd.read_csv('../data/stsbenchmark/train.csv')

# sentence1, sentence2, labels = preprocess(train_data)
# X1 = torch.tensor(sentence1)
# X2 = torch.tensor(sentence2)
# y = torch.tensor(labels)

# torch.save(X1, 'X1.pt')
# torch.save(X2, 'X2.pt')
# torch.save(y, 'y.pt')

# torch.save(X1, 'X1_sick.pt')
# torch.save(X2, 'X2_sick.pt')
# torch.save(y, 'y_sick.pt')

X1 = torch.load('X1.pt')
X2 = torch.load('X2.pt')
y = torch.load('y.pt')

# X1 = torch.load('X1_sick.pt')
# X2 = torch.load('X2_sick.pt')
# y = torch.load('y_sick.pt')

X1_train = X1.float()
X2_train = X2.float()
y_train = y.float()

#### STS Benchmark

# val_data = pd.read_csv('../data/stsbenchmark/validation.csv')

# sentence1, sentence2, labels = preprocess(val_data)
# X1_val = torch.tensor(sentence1).float()
# X2_val = torch.tensor(sentence2).float()
# y_val = torch.tensor(labels).float()

# torch.save(X1_val, 'X1_val.pt')
# torch.save(X2_val, 'X2_val.pt')
# torch.save(y_val, 'y_val.pt')

X1_val = torch.load('X1_val.pt')
X2_val = torch.load('X2_val.pt')
y_val = torch.load('y_val.pt')

X1_train = torch.cat((X1_train, X1_val), dim=0)
X2_train = torch.cat((X2_train, X2_val), dim=0)
y_train = torch.cat((y_train, y_val), dim=0)

n_samples = X1_train.size(0)
n_train = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)
train_indices, val_indices = indices[:n_train], indices[n_train:]
X1_train, X1_val = X1_train[train_indices], X1_train[val_indices]
X2_train, X2_val = X2_train[train_indices], X2_train[val_indices]
y_train, y_val = y_train[train_indices], y_train[val_indices]


train = data_utils.TensorDataset(X1_train, X2_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True)

val = data_utils.TensorDataset(X1_val, X2_val, y_val)
val_loader = data_utils.DataLoader(val, batch_size=32, shuffle=True)


model = Model()
train_loss, val_loss, model = train_model(model, train_loader, val_loader, epochs=10, lr=0.001)
torch.save(model, 'model_2.pt')