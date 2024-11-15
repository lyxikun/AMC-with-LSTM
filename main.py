# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import algorithm as al
import dataprocess as dp
import torch
import torch.nn as nn
import matplotlib.pyplot as pl

from torchsummary import summary

MODE = 1
MODULEPATH = './models/LSTM.pkl'

if torch.cuda.is_available():
    print("GPU Available")
else:
    print("No GPU Available")

save_dir = 'models/LSTM.pkl'

modulationtype = ['8PSK', 'QPSK', 'QAM16', 'QAM64']
snrtype = [6,8,12,14,18]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 准备数据
    input_size = 2  # 输入特征数 列数
    hidden_size = 16  # 隐藏层特征数
    num_layers = 3 # LSTM层数
    output_size = 11  # 输出类别数
    batch_size = 400  # 批大小  行数
    sequence_length = 128  # 序列长度 time step 维度

    # x = torch.randn(sequence_length, batch_size, input_size)
    # y = torch.FloatTensor([1,0,0,0,0,0])

    #print(x)
    model = al.LSTM(input_size, hidden_size, num_layers, output_size, True)

    #Using CrossEntropy theory as lossfunc
    criterion = nn.CrossEntropyLoss()
    #optimizer using adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    cuda = next(model.parameters()).device
    train_loader, test_loader = dp.load_data(batch_size, cuda, modulationtype, snrtype)

    # 假设模型实例化为 model，输入数据的维度为 (batch_size, sequence_length, input_size)
    # input_size = (sequence_length, input_size)  # 输入维度 (不包括 batch_size)
    # summary(model, input_size=input_size, device="cuda" if torch.cuda.is_available() else "cpu")

    if MODE:
        print('Start training...')

        # 开始训练
        num_epochs = 450
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (inputs, labels) in enumerate(train_loader, 0):
                outputs = model(inputs.mT)
                loss = criterion(outputs, labels[:,0].to(torch.long))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels[:,0]).sum().item()
                total_predictions += labels[:,0].size(0)


            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions * 100
            scheduler.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, lr: {optimizer.param_groups[0]['lr']:.5f}')


        torch.save(model, save_dir)

        print('Training finished.')

    # 预测新数据
    if MODE == 0:
        model = torch.load(MODULEPATH)

    with torch.no_grad():
        print('Start predicting...')

        model.eval()
        correct_predictions = 0
        total_predictions = 0
        test_loss = 0
        for i, (inputs, labels) in enumerate(test_loader, 0):
            outputs = model(inputs.mT)
            loss = criterion(outputs, labels[:,0].to(torch.long))

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels[:,0]).sum().item()
            total_predictions += labels[:,0].size(0)
            test_loss += loss.item()

        avg_loss = test_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100

        print(f'Test: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')