import torch
import numpy as np
import pandas as pd
from cnn_utils import preprocess

# test_data = pd.read_csv('../data/sick/test.csv')

# senteces_A = test_data['sentence_A'].values
# sentences_B = test_data['sentence_B'].values
# labels = test_data['normalised_score'].values

test_data = pd.read_csv('../data/stsbenchmark/test.csv')

senteces_A = test_data['sentence1'].values
sentences_B = test_data['sentence2'].values
labels = test_data['similarity'].values

# sentence1, sentence2, labels = preprocess(test_data)
# X1_test = torch.tensor(sentence1).float()
# X2_test = torch.tensor(sentence2).float()
# y_test = torch.tensor(labels).float()

# torch.save(X1_test, 'X1_test_sick.pt')
# torch.save(X2_test, 'X2_test_sick.pt')
# torch.save(y_test, 'y_test_sick.pt')

# torch.save(X1_test, 'X1_test.pt')
# torch.save(X2_test, 'X2_test.pt')
# torch.save(y_test, 'y_test.pt')

X1_test = torch.load('X1_test.pt')
X2_test = torch.load('X2_test.pt')
y_test = torch.load('y_test.pt')

# X1_test = torch.load('X1_test_sick.pt')
# X2_test = torch.load('X2_test_sick.pt')
# y_test = torch.load('y_test_sick.pt')

model = torch.load('model_2.pt')

model.eval()

with torch.no_grad():
    y_pred = model(X1_test, X2_test)
    y_pred = y_pred.numpy()
    y_pred = np.squeeze(y_pred)
    y_pred = np.clip(y_pred, 0, 5)
    results = pd.DataFrame({'sentence1': senteces_A, 'sentence2': sentences_B, 'similarity': labels, 'predicted_similarity': y_pred})
    results.to_csv('results/results_cnn_sts.csv', index=False)
    print('Predictions saved in results_cnn_sts.csv')

correlation = pd.Series(labels).corr(pd.Series(y_pred))
print('Correlation between expected and predicted similarity scores:', correlation)