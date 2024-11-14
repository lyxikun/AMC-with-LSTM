# This is a sample Python script.
from tensorboard.backend.event_processing.event_file_inspector import print_dict

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import algorithm as al
import dataprocess as dp
import torch
import torch.nn as nn

MODE = 1
MODULEPATH = './models/LSTM-4-100.pkl'

if torch.cuda.is_available():
    print("GPU Available")
else:
    print("No GPU Available")

save_dir = 'models/LSTM.pkl'

modulationtype = ['8PSK', 'BPSK']
snrtype = [8,12,14]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 准备数据
    input_size = 2  # 输入特征数 列数
    hidden_size = 16  # 隐藏层特征数
    num_layers = 1 # LSTM层数
    output_size = 2  # 输出类别数
    batch_size = 480  # 批大小  行数
    sequence_length = 10  # 序列长度 time step 维度

    # x = torch.randn(sequence_length, batch_size, input_size)
    # y = torch.FloatTensor([1,0,0,0,0,0])

    #print(x)
    model = al.LSTM(input_size, hidden_size, num_layers, output_size)
    #Using CrossEntropy theory as lossfunc
    criterion = nn.CrossEntropyLoss()
    #optimizer using adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    cuda = next(model.parameters()).device
    train_loader, test_loader = dp.load_data(batch_size, cuda, modulationtype, snrtype)

    if MODE:
        print('Start training...')

        # 开始训练
        num_epochs = 1000
        num_batches = 10
        model.train()
        for epoch in range(num_epochs):
            sum_loss = 0
            loss = 0
            PT = 0
            PF = 0
            for i, data in enumerate(train_loader, 0):
                if i == num_batches:
                    break
                inputs, labels = data
                for batches in range(batch_size):
                    outputs = model(inputs[batches].transpose(0,1))
                    _, predicted = torch.max(outputs.data, 0)
                    loss = criterion(outputs, labels[batches])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Accuracy
                    # if labels[batches][predicted.item()].item() == 1:
                    if labels[batches] == predicted.item():
                        PT += 1
                    else:
                        PF += 1
                    sum_loss += loss.item()
                # if (i + 1) % 8 == 0:
                #     print('Batch epoch [{}/{}], Loss: {:.4f}'.format(i + 1, 128, avg_loss/(i+1)))
            # if (epoch+1) % 10 == 0:
            #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, sum_loss/(batch_size*num_batches), PT/(PT+PF)))

        torch.save(model, save_dir)

        print('Training finished.')

    # 预测新数据
    if MODE == 0:
        model = torch.load(MODULEPATH)
    num_tests = 400
    with torch.no_grad():
        print('Start predicting...')
        PT = 0
        PF = 0
        sumloss = 0
        model.eval()
        for test in range(num_tests):
            for i, data in enumerate(test_loader, 0):
                if i == 10:
                    break
                inputs, labels = data
                outputs = model(inputs[i].transpose(0,1))
                _, predicted = torch.max(outputs.data, 0)
                loss = criterion(outputs, labels[i])
                sumloss += loss.item()
                if labels[i] == predicted.item():
                    PT += 1
                else:
                    PF += 1

        print('Accuracy: {:.4f}, Loss: {:.4f}'.format(PT/(PT+PF), sumloss/(128*num_tests)))
