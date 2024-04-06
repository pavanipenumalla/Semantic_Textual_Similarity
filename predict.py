from score import get_similarity_score
import pandas as pd

data = pd.read_csv('data/sick/test.csv')

sentences_A = data['sentence_A'].values
sentences_B = data['sentence_B'].values
scores = data['normalised_score'].values

predicted_scores = []
for i in range(len(sentences_A)):
    predicted_scores.append(get_similarity_score(sentences_A[i], sentences_B[i]))

results = pd.DataFrame({'sentence_A': sentences_A, 'sentence_B': sentences_B, 'expected_score': scores, 'predicted_score': predicted_scores})
results.to_csv('results/sick/results.csv', index=False)