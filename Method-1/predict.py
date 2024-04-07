from score import get_similarity_score
import pandas as pd

data = pd.read_csv('data/stsbenchmark/test.csv')

# modify according to dataset
sentences_A = data['sentence1'].values
sentences_B = data['sentence2'].values
scores = data['similarity'].values

n_values = [1,2,3,4,5]
for n in n_values:
    print('Predicting for n = {}'.format(n))
    predicted_scores = []
    sent1 = []
    sent2 = []
    exp_scores = []
    for i in range(len(sentences_A)):
        predicted_score = get_similarity_score(sentences_A[i], sentences_B[i],n)
        if predicted_score == -1:
            continue
        predicted_scores.append(predicted_score)
        sent1.append(sentences_A[i])
        sent2.append(sentences_B[i])
        exp_scores.append(scores[i])

    results = pd.DataFrame({'sentence_A': sent1, 'sentence_B': sent2 ,'expected_score': exp_scores, 'predicted_score': predicted_scores})
    results.to_csv('results/stsbenchmark/results_{}.csv'.format(n), index=False)