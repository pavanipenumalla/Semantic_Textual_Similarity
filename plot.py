import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/sick/SICK.csv')

plt.figure(figsize=(10, 6))
plt.bar(data['sentence_A'] + ' & ' + data['sentence_B'], data['normalised_score'])
plt.xlabel('Sentence Pairs')
plt.ylabel('Relatedness Score')
plt.title('Relatedness Score for Sentence Pairs')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
