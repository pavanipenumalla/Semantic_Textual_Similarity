import torch
import numpy as np
import pandas as pd
from bert_utils import preprocess_data, concatenate_data, tokenize_data, get_sentence_embeddings
import sys
from torch.utils.data import DataLoader, TensorDataset

def main(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, _, _, _, _, test_data_sentence1, test_data_sentence2, test_data_scores = preprocess_data(data)

    test_tokenized_sent1, tokenizer = tokenize_data(test_data_sentence1)
    test_tokenized_sent2, tokenizer = tokenize_data(test_data_sentence2)

    test_input_ids, test_segment_ids, test_attention_masks = concatenate_data(test_tokenized_sent1, test_tokenized_sent2, tokenizer)

    test_sentence_embeddings = get_sentence_embeddings(test_input_ids, test_segment_ids, test_attention_masks)    

    model = torch.load(f'models/model_{data}.pt')
    model.to(device)

    if data == 'sts':
        test_data = pd.read_csv('../data/sts/test.csv')
        senteces_A = test_data['sentence1'].values
        sentences_B = test_data['sentence2'].values
        labels = test_data['similarity'].values
    elif data == 'sick':
        test_data = pd.read_csv('../data/sick/test.csv')
        senteces_A = test_data['sentence_A'].values
        sentences_B = test_data['sentence_B'].values
        labels = test_data['normalised_score'].values

    test_sentence_embeddings = torch.tensor(test_sentence_embeddings).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(test_sentence_embeddings)
        y_pred = y_pred.cpu()
        y_pred = y_pred.numpy()
        y_pred = np.squeeze(y_pred)
        y_pred = np.clip(y_pred, 0, 5)
        results = pd.DataFrame({'sentence1': senteces_A, 'sentence2': sentences_B, 'similarity': labels, 'predicted_similarity': y_pred})
        results.to_csv(f'results/results_bert_{data}.csv', index=False)
        print(f'Predictions saved in results_bert_{data}.csv')

    correlation = pd.Series(labels).corr(pd.Series(y_pred))
    print('Correlation between expected and predicted similarity scores:', correlation)

if __name__ == '__main__':
    data = sys.argv[1]
    main(data)