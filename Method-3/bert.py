from bert_utils import preprocess_data, concatenate_data, tokenize_data, get_sentence_embeddings, train_FCNN, FCNN
import sys
import torch
import torch.nn as nn

def main(data):
    train_data_sentence1, train_data_sentence2, train_data_scores, val_data_sentence1, val_data_sentence2, val_data_scores, test_data_sentence1, test_data_sentence2, test_data_scores = preprocess_data('sts')

    train_tokenized_sent1, tokenizer = tokenize_data(train_data_sentence1)
    train_tokenized_sent2, tokenizer = tokenize_data(train_data_sentence2)

    val_tokenized_sent1, tokenizer = tokenize_data(val_data_sentence1)
    val_tokenized_sent2, tokenizer = tokenize_data(val_data_sentence2)

    train_input_ids, train_segment_ids, train_attention_masks = concatenate_data(train_tokenized_sent1, train_tokenized_sent2, tokenizer)
    val_input_ids, val_segment_ids, val_attention_masks = concatenate_data(val_tokenized_sent1, val_tokenized_sent2, tokenizer)

    train_sentence_embeddings = get_sentence_embeddings(train_input_ids, train_segment_ids, train_attention_masks)
    val_sentence_embeddings = get_sentence_embeddings(val_input_ids, val_segment_ids, val_attention_masks)

    fcnn_input_size = train_sentence_embeddings.shape[1]
    fcnn_hidden_size = 256
    fcnn_output_size = 1
    fcnn_activation = 'relu'
    fcnn_num_layers = 2
    model = FCNN(fcnn_input_size, fcnn_hidden_size, fcnn_output_size, fcnn_activation, fcnn_num_layers)

    lr = 0.001
    n_epochs = 10
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model,train_losses, val_losses = train_FCNN(model, criterion, train_sentence_embeddings, train_data_scores, val_sentence_embeddings, val_data_scores, n_epochs, batch_size, device, lr)

    torch.save(model, f'models/model_{data}.pt')

if __name__ == '__main__':
    data = sys.argv[1]
    main(data)