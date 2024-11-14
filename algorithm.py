import torch
import torch.nn as nn
from torch.nn.functional import softmax


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.fc6 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.log2s = nn.LogSoftmax(dim=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        h0 = torch.nn.init.normal_(h0, mean=0, std=1)
        c0 = torch.nn.init.normal_(c0, mean=0, std=1)
        out, (h0, c0) = self.lstm(x, (h0, c0))
        #out = self.fc4(out)
        #out = torch.atan(out[0][0] / out[0][1])
        #out = self.bn(out)
        # Attention Mechanism
        # alpha = self.fc6(out)
        # alpha = self.softmax(alpha)
        # out = alpha * out
        out = out[:,-1,:]
        # Fullconnect layer
        out = self.fc2(out)
        out = self.relu(out)

        # out = self.fc3(out)
        # out = self.relu(out)
        # #
        # out = self.fc4(out)
        # out = self.relu(out)

        # out = self.fc2(out)
        # out = self.relu(out)
        #
        # out = self.fc2(out)
        # out = self.relu(out)

        out = self.fc5(out)

        return out