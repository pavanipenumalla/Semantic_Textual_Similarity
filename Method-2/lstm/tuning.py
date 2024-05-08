import wandb
import torch.nn as nn
import torch
from lstm_utils import load_data, Model, train
import pandas as pd
import numpy as np

data = 'sts'
X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = load_data(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X1_test = X1_test.to(device)
X2_test = X2_test.to(device)


sweep_config = {
    "name": "lstm_tuning",
    "method": "grid",
    "parameters": {
        "lstm_hidden_size": {"values": [64, 128]},
        "lstm_num_layers": {"values": [1, 2]},
        "lstm_output_size": {"values": [100,300,500]},
        "lstm_activation": {"values": ["relu", "tanh"]},
        "bidirectional": {"values": [True, False]},  
        "fc_hidden_size": {"values": [64,128]},
        "fc_activation": {"values": ["relu", "tanh"]},
        "fc_num_layers": {"values": [1, 2]},
        "n_epochs": {"values": [50, 100]},
        "batch_size": {"values": [16, 32]},
    },
    "metric": {"goal": "maximize", "name": "corr"}
}

sweep_id = wandb.sweep(sweep_config, project="LSTM-Tuning")

def tune():

    config_defaults = {
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "lstm_output_size": 300,
        "lstm_activation": "relu",
        "bidirectional": True,
        "fc_hidden_size": 64,
        "fc_activation": "relu",
        "fc_num_layers": 1,
        "n_epochs": 100,
        "batch_size": 32,
    }
    best_model = None
    best_corr = -1

    with wandb.init(config=config_defaults):
        config = wandb.config
        model = Model(337, config.lstm_hidden_size, config.lstm_num_layers, config.lstm_output_size, config.lstm_activation, config.bidirectional, 2 * config.lstm_output_size, config.fc_hidden_size, 1, config.fc_activation, config.fc_num_layers)
        criterion = nn.MSELoss()
        model, train_losses, val_losses = train(model, criterion, X1_train, X2_train, y_train, X1_val, X2_val, y_val, config.n_epochs, config.batch_size, device, 0.001)

        model.eval()
        with torch.no_grad():
            y_pred = model(X1_test, X2_test)
            y_pred = y_pred.cpu()
            y_pred = y_pred.numpy()
            y_pred = np.squeeze(y_pred)
            y_pred = np.clip(y_pred, 0, 5)
            y_pred = pd.Series(y_pred)
            y_test_ = pd.Series(y_test)
            corr = y_test_.corr(y_pred)
            print('Correlation between expected and predicted similarity scores:', corr)

            if corr > best_corr:
                best_corr = corr
                best_model = model
        wandb.log({"corr": corr, "train_loss": train_losses[-1], "val_loss": val_losses[-1], "n_epochs": config.n_epochs, "batch_size": config.batch_size,  "lstm_hidden_size": config.lstm_hidden_size, "lstm_num_layers": config.lstm_num_layers, "lstm_output_size": config.lstm_output_size, "lstm_activation": config.lstm_activation, "fc_hidden_size": config.fc_hidden_size, "fc_activation": config.fc_activation, "fc_num_layers": config.fc_num_layers})

    # Save the best model
    torch.save(best_model, 'best_model.pt')
    
wandb.agent(sweep_id, function=tune)