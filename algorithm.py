import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_a = nn.Parameter(torch.randn(hidden_size, 1))  # 定义注意力权重矩阵
        self.b_a = nn.Parameter(torch.zeros(1))  # 注意力偏置

    def forward(self, h):
        # h 的形状为 (batch_size, sequence_length, hidden_size)

        # 计算注意力得分 (batch_size, sequence_length)
        score = torch.matmul(h, self.W_a) + self.b_a
        score = torch.sigmoid(score).squeeze(-1)  # 应用激活函数 sigma，形状为 (batch_size, sequence_length)

        # 归一化得分 (batch_size, sequence_length)
        alpha = nn.functional.softmax(score, dim=1)

        return alpha  # 返回注意力权重

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, ifatt):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.attention = Attention(hidden_size)
        self.ifatt = ifatt
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #权重正态分布
        h0 = torch.nn.init.normal_(h0, mean=0, std=1)
        c0 = torch.nn.init.normal_(c0, mean=0, std=1)
        out, (h0, c0) = self.lstm(x, (h0, c0))

        if self.ifatt:
            alpha = self.attention(out)
            alpha = alpha.unsqueeze(-1)
            out = torch.sum(alpha * out, dim=1)
            out = self.relu(out)
        else:
            out = self.relu(out[:,-1,:])

        out = self.fc1(out)

        out = self.relu(out)
        out = self.fc2(out)

        out = self.relu(out)
        out = self.fc4(out)

        return out