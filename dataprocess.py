import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def load_data(batch_size, device, modulationtype, snrtype):
    # 加载RML2016.10a数据集
    file_path = './datasets/RML2016.10a_dict.pkl'  # 修改为数据集的实际路径
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # 存储信号和标签的列表
    signals = []
    labels = []

    # 遍历数据集字典，提取符合条件的信号和标签
    for (mod_type, snr), samples in data.items():
        #if mod_type in modulationtype and snr in snrtype:
            signals.append(samples)  # 添加符合条件的信号数据
            labels.extend([(mod_type, snr)] * samples.shape[0])  # 为每个信号样本添加标签

    # 将信号和标签转换为 NumPy 数组
    signals = np.vstack(signals)  # (总样本数, 2, 序列长度)
    labels = np.array(labels)  # (总样本数, 2)

    x = torch.linspace(0,127,128)

    # 编码调制类型标签
    le = LabelEncoder()
    modulation_labels = le.fit_transform(labels[:, 0])  # 编码调制类型

    # 组合最终标签 (modulation_label, snr)
    final_labels = np.column_stack((modulation_labels, labels[:, 1].astype(np.float32)))

    train_signals, test_signals, train_labels, test_labels = train_test_split(
        signals, final_labels, test_size=0.2, random_state=42
    )

    class RMLDatasetWithNormalization(Dataset):
        def __init__(self, signals, labels, target_energy=signals.shape[0]):
            self.signals = torch.tensor(signals, dtype=torch.float32).to(device)
            self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
            self.target_energy = target_energy

        def __len__(self):
            return len(self.signals)

        def normalize_energy(self, signal):
            # 计算信号的能量
            energy = torch.sum(signal ** 2)
            # 计算归一化因子
            norm_factor = torch.sqrt(self.target_energy / energy)
            # 应用归一化因子
            return signal * norm_factor

        def __getitem__(self, idx):
            signal = self.signals[idx]
            label = self.labels[idx]
            # 对信号进行能量归一化
            signal = self.normalize_energy(signal)
            return signal, label

    # 创建 DataLoader
    train_dataset = RMLDatasetWithNormalization(train_signals, train_labels)
    test_dataset = RMLDatasetWithNormalization(test_signals, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



