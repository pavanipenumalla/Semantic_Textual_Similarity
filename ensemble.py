import pandas as pd
import numpy as np

df1 = pd.read_csv('Method-2/cnn/results/results_cnn_sts.csv')
cnn_scores = df1['predicted_similarity'].values
original_scores = df1['similarity'].values

df2 = pd.read_csv('Method-2/lstm/results/results_lstm_sts.csv')
lstm_scores = df2['predicted_similarity'].values

df3 = pd.read_csv('Method-2/rnn/results/results_rnn_sts.csv')
rnn_scores = df3['predicted_similarity'].values

df4 = pd.read_csv('Method-3/results/results_bert_sts.csv')
bert_scores = df4['predicted_similarity'].values

ensemble_scores = (cnn_scores + lstm_scores + rnn_scores + bert_scores) / 4

corr = pd.Series(original_scores).corr(pd.Series(ensemble_scores))
print('Correlation:', corr)