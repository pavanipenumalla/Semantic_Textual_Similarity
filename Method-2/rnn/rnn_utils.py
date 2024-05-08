import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def load_data(data):
    directory = f'../preprocessed_data/{data}/'
    if data == 'sts':
        X1 = torch.load(directory + 'X1.pt')
        X2 = torch.load(directory + 'X2.pt')
        y = torch.load(directory + 'y.pt')
        X1_val = torch.load(directory + 'X1_val.pt')
        X2_val = torch.load(directory + 'X2_val.pt')
        y_val = torch.load(directory + 'y_val.pt')
        X1_train = X1.float()
        X2_train = X2.float()
        y_train = y.float()
        X1_val = X1_val.float()
        X2_val = X2_val.float()
        y_val = y_val.float()
        X1_test = torch.load(directory + 'X1_test.pt')
        X2_test = torch.load(directory + 'X2_test.pt')
        y_test = torch.load(directory + 'y_test.pt')
        X1_test = X1_test
        X2_test = X2_test
        y_test = y_test
        return X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test
    elif data == 'sick':
        X1_train = torch.load(directory + 'X1_sick.pt')
        X2_train = torch.load(directory + 'X2_sick.pt')
        y_train = torch.load(directory + 'y_sick.pt')
        n_samples = X1_train.size(0)
        n_train = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_indices, val_indices = indices[:n_train], indices[n_train:]
        X1_train, X1_val = X1_train[train_indices], X1_train[val_indices]
        X2_train, X2_val = X2_train[train_indices], X2_train[val_indices]
        y_train, y_val = y_train[train_indices], y_train[val_indices]
        X1_test = torch.load(directory + 'X1_test_sick.pt')
        X2_test = torch.load(directory + 'X2_test_sick.pt')
        y_test = torch.load(directory + 'y_test_sick.pt')
        return X1_train.float(), X2_train.float(), y_train.float(), X1_val.float(), X2_val.float(), y_val.float(), X1_test.float(), X2_test.float(), y_test.float()
    
# RNN model architecture
# input -> batch_size, seq_len, input_size -> [batch_size, 30, 300]
# output -> batch_size, 300
# activation -> ReLU
# no. of layers -> 1

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        return out

# FCNN model architecture
# input -> batch_size, 600
# output -> batch_size, 1

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation, num_layers=1):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.num_layers = num_layers
        for i in range(num_layers-1):
            self.fc = nn.Linear(hidden_size, hidden_size)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for i in range(self.num_layers-1):
            x = self.fc(x)
            x = self.activation(x)
        x = self.fc2(x)
        return x
    
# model
# first sent = batch_size,30,300 -> RNN model -> batch_size,300
# second sent = batch_size,30,300 -> RNN model -> batch_size,300
# concat element wise diff and element wise product
# concat the above two -> batch_size,600
# feed to a fully connected layer -> batch_size,1

class Model(nn.Module):
    def __init__(self, rnn_input_size, rnn_hidden_size, rnn_num_layers, rnn_output_size, rnn_activation, fc_input_size, fc_hidden_size, fc_output_size, fc_activation, fc_num_layers):
        super(Model, self).__init__()
        self.rnn = RNN(rnn_input_size, rnn_hidden_size, rnn_num_layers, rnn_output_size, rnn_activation)
        self.fc = FCNN(fc_input_size, fc_hidden_size, fc_output_size, fc_activation, fc_num_layers)

    def forward(self, x1, x2):
        x1 = self.rnn(x1)
        x2 = self.rnn(x2)
        x = torch.cat((x1, x2), 1)
        x_diff = torch.abs(x1 - x2)
        x_prod = x1 * x2
        x = torch.cat((x_diff, x_prod), 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x
    
# train function
# input -> model, criterion, optimizer, X1_train, X2_train, y_train, X1_val, X2_val, y_val, num_epochs, batch_size, device, learning_rate
# output -> model, train_loss, val_loss

def train(model, criterion, X1_train, X2_train, y_train, X1_val, X2_val, y_val, num_epochs, batch_size, device, learning_rate):

    train_data = TensorDataset(X1_train, X2_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TensorDataset(X1_val, X2_val, y_val)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model.to(device)

    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            x1, x2, labels = data
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
    
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss.append(running_loss / len(train_loader))
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss / len(train_loader)))

        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            for i, data in enumerate(val_loader):
                x1, x2, labels = data
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                outputs = model(x1, x2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

        val_loss.append(val_running_loss / len(val_loader))
        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, num_epochs, val_running_loss / len(val_loader)))

    return model, train_loss, val_loss