import pandas as pd

data = pd.read_csv('results/sick/results.csv')

expected_scores = data['expected_score'].values
predicted_scores = data['predicted_score'].values

correlation = pd.Series(expected_scores).corr(pd.Series(predicted_scores))
print(correlation)