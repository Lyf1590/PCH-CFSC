import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn import functional as F


class img_dim_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(img_dim_layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, output_dim)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class txt_dim_layer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(txt_dim_layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, output_dim)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MDE(nn.Module):
    def __init__(self, hidden_dim=[512, 2048, 1024, 128], act=nn.Tanh(), dropout=0.1):
        super(MDE, self).__init__()
        # 设置了模型的输入维度、隐藏层维度和输出维度
        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim[-1]
        # 用于存储模型的各个层、激活函数和批归一化层
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  

        # Add input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        self.activations.append(act)
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[0]))

        # Add hidden layers 使用循环遍历隐藏层的维度列表，为每一层创建线性层、激活函数和批归一化层
        for i in range(len(self.hidden_dim) - 1):
            self.layers.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            self.activations.append(act)
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[i + 1])) 

        # Add output layer 128->128 linear
        self.layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
        # 定义丢弃层，用于在模型的训练过程中进行随机丢弃操作
        self.dropout = nn.Dropout(p=dropout)  # p丢弃的概率

    def forward(self, x):
        for layer, activation, batch_norm in zip(self.layers, self.activations, self.batch_norms):
            x = layer(x)
            x = activation(x)
            x = batch_norm(x)
            x = self.dropout(x)

        return x


class ImageMlp(nn.Module):                                  # [2048, 1024, 512]
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 512, 256], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim_feedforward[0])
        self.fc2 = nn.Linear(dim_feedforward[0], dim_feedforward[1])
        self.fc3 = nn.Linear(dim_feedforward[1], dim_feedforward[2])
        self.fc4 = nn.Linear(dim_feedforward[2], hash_lens)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, X):
        x = self.relu(self.fc1(X))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.tanh(x)
        return x



class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 512, 256], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim_feedforward[0])
        self.fc2 = nn.Linear(dim_feedforward[0], dim_feedforward[1])
        self.fc3 = nn.Linear(dim_feedforward[1], dim_feedforward[2])
        self.fc4 = nn.Linear(dim_feedforward[2], hash_lens)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, X):
        x = self.relu(self.fc1(X))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.tanh(x)
        return x
