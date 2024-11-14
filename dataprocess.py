import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

def load_data(batch_size, device, modulationtype, snrtype):
    # 加载RML2016.10a数据集
    file_path = './datasets/RML2016.10a_dict.pkl'  # 修改为数据集的实际路径
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # 提取数据和标签
    X = []
    Y = []
    modulations = list(set([mod for mod, snr in data.keys()]))
    for mod, snr in data.keys():
        for mod_needed in modulationtype:
            if mod == mod_needed:
                for snr_needed in snrtype:
                    if snr == snr_needed:
                        samples = data[(mod, snr)]
                        #label = modulations.index(mod)  # 将调制类型转换为数值标签
                        label = modulationtype.index(mod)
                        for sample in samples:
                            X.append(sample)
                            Y.append(label)

    # 转换为PyTorch张量
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.long)

    # 归一化
    # energy = torch.sum(X ** 2, dim=(1, 2), keepdim=True)
    # X = X / torch.sqrt(energy + 1e-6)  # 归一化，使得每个样本的能量为1

    X = X.transpose(1,2)
    # 将标签转换为独热编码
    num_classes = len(modulationtype)
    Y_one_hot = F.one_hot(Y, num_classes=num_classes)
    Y_one_hot = Y

    if torch.cuda.is_available():
        X = X.to(device)
        Y = Y.to(device)
        Y_one_hot = Y_one_hot.to(device)

    # 计算训练集和测试集的大小 (80% 训练, 20% 测试)
    total_samples = len(X)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    # 定义自定义数据集类
    class RML2016Dataset(Dataset):
        def __init__(self, X, Y_one_hot):
            self.X = X.unsqueeze(1)  # 增加一个维度，形状变为 (N, 1, 2, 128)
            self.Y = Y_one_hot

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    # 使用random_split划分数据集
    train_dataset, test_dataset = random_split(
        RML2016Dataset(X, Y_one_hot), [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for i, data in enumerate(test_loader, 0):
    #     inputs, labels = data
    #     print(inputs.shape)
    #     print(labels.shape)
    #     break
    #a=1

    return train_loader, test_loader
