import torch
import torch.nn as nn

class CNNTrendNet(nn.Module):

    def __init__(self, state_dim, model_cfg):
        super().__init__()
        timestep_dim, feature_dim = state_dim
        num_filter = model_cfg.num_filter
        action_dim = 2


        self.conv1 = nn.Conv2d(1, num_filter, kernel_size=(1, feature_dim))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=(3, 1))
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(num_filter, num_filter, kernel_size=(3, 1))
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(int((((timestep_dim - 3 + 1) // 2)    - 3 + 1) // 2) * num_filter, action_dim)


    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x
    

class LSTMTrendNet(nn.Module):
    def __init__(self, state_dim, model_cfg):
        super().__init__()
        timestep_dim, feature_dim = state_dim
        hidden_size = model_cfg.hidden_size
        action_dim = 2

        self.lstm1 = nn.LSTM(feature_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        lstm_o, _ = self.lstm1(x)
        lstm_final_o = lstm_o[:, -1, :]
        lstm_final_o = lstm_final_o.view(lstm_final_o.shape[0], -1)
        fc1_o = self.fc1(lstm_final_o)
        relu_o = self.relu1(fc1_o)
        out = self.fc2(relu_o)
        return out
    

class ALSTMTrendNet(nn.Module):
    def __init__(self, state_dim, model_cfg):
        super().__init__()
        timestep_dim, feature_dim = state_dim
        hidden_size = model_cfg.hidden_size
        action_dim = 2

        self.lstm1 = nn.LSTM(feature_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        lstm_o, _ = self.lstm1(x)
        lstm_final_o = lstm_o[:, -1, :]
        lstm_final_o = lstm_final_o.view(lstm_final_o.shape[0], -1)
        fc1_o = self.fc1(lstm_final_o)
        relu_o = self.relu1(fc1_o)
        out = self.fc2(relu_o)
        return out