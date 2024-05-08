import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def preprocess_data(data):
    directory = f'../data/{data}/'
    if data == 'sts':
        train_data = pd.read_csv(directory + 'train.csv')
        val_data = pd.read_csv(directory + 'validation.csv')
        test_data = pd.read_csv(directory + 'test.csv')
        train_data_sentence1 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in train_data['sentence1'].values]
        train_data_sentence2 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in train_data['sentence2'].values]
        train_data_scores = train_data['similarity'].values
        val_data_sentence1 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in val_data['sentence1'].values]
        val_data_sentence2 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in val_data['sentence2'].values]
        val_data_scores = val_data['similarity'].values
        test_data_sentence1 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in test_data['sentence1'].values]
        test_data_sentence2 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in test_data['sentence2'].values]
        test_data_scores = test_data['similarity'].values
        return train_data_sentence1, train_data_sentence2, train_data_scores, val_data_sentence1, val_data_sentence2, val_data_scores, test_data_sentence1, test_data_sentence2, test_data_scores
    elif data == 'sick':
        train_val_data = pd.read_csv(directory + 'train.csv')
        test_data = pd.read_csv(directory + 'test.csv')
        n_samples = train_val_data.shape[0]
        n_train = int(0.8 * n_samples)
        indices = list(range(n_samples))
        random.shuffle(indices)
        train_indices, val_indices = indices[:n_train], indices[n_train:]
        train_data = train_val_data.iloc[train_indices]
        val_data = train_val_data.iloc[val_indices]
        train_data_sentence1 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in train_data['sentence_A'].values]
        train_data_sentence2 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in train_data['sentence_B'].values]
        train_data_scores = train_data['normalised_score'].values
        val_data_sentence1 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in val_data['sentence_A'].values]
        val_data_sentence2 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in val_data['sentence_B'].values]
        val_data_scores = val_data['normalised_score'].values
        test_data_sentence1 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in test_data['sentence_A'].values]
        test_data_sentence2 = [sentence + '.' if sentence[-1]!='.' else sentence for sentence in test_data['sentence_B'].values]
        test_data_scores = test_data['normalised_score'].values
        return train_data_sentence1, train_data_sentence2, train_data_scores, val_data_sentence1, val_data_sentence2, val_data_scores, test_data_sentence1, test_data_sentence2, test_data_scores

def tokenize_data(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    return tokenized_sentences, tokenizer

def concatenate_data(tokenized_sent1, tokenized_sent2, tokenizer):
    input_ids = []
    segment_ids = []
    attention_masks = []
    max_len = max([len(tokenized_sent1[i]) + len(tokenized_sent2[i]) + 3 for i in range(len(tokenized_sent1))])
    for i in range(len(tokenized_sent1)):
        encoded_dict = tokenizer.encode_plus(tokenized_sent1[i], tokenized_sent2[i], add_special_tokens=True, max_length=max_len, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True)
        input_ids.append(encoded_dict['input_ids'])
        segment_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, segment_ids, attention_masks

def get_sentence_embeddings(input_ids, segment_ids, attention_masks):
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_masks)
        hidden_states = outputs[2]
        hidden_states = hidden_states[-1:]
        word_embeddings = torch.stack(hidden_states, dim=0).sum(dim=0)
        sentence_embeddings = word_embeddings.mean(dim=1)
    return sentence_embeddings

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

def train_FCNN(model, criterion, X_train, Y_train, X_val, Y_val, n_epochs, batch_size, device, learning_rate):

    train_data = TensorDataset(X_train, torch.tensor(Y_train, dtype=torch.float))
    val_data = TensorDataset(X_val, torch.tensor(Y_val, dtype=torch.float))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            val_losses.append(running_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}')

    return model, train_losses, val_losses
    