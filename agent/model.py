import torch
import torch.nn as nn

class TrendNet(nn.Module):

    def __init__(self, state_dim, model_cfg):
        super().__init__()
        timestep_dim, feature_dim = state_dim
        num_filter = model_cfg.num_filter
        action_dim = model_cfg.action_dim


        self.conv1 = nn.Conv2d(1, num_filter, kernel_size=(1, feature_dim))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=(3, 1))
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(int((timestep_dim - 3 + 1) // 2) * num_filter, action_dim)


    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x
