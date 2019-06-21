from keras.utils import plot_model
from keras.models import load_model
from autokeras.utils import pickle_from_file
from keras.datasets import mnist
from autokeras import ImageClassifier
from autokeras import supervised
from autokeras.constant import Constant
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import copy
import os


# 在测试集上的准确率
def acc_val(net, testloader):
    net.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).item()
    print("accuracy of the network on the 10000 test images: {:.3f}".format(correct / total))
    net.train()

# 在每个类别上的准确率
def acc_val_classes(net, testloader, classes):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            label = labels[0]
            class_correct[label] += (predicted == labels).item()
            class_total[label] += 1
    net.train()
    for i in range(10):
        print("accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))


# model = load_model('./my_model.h5')


model_file_name = './my_model.h5'
model = pickle_from_file(model_file_name)
# results = model.evaluate(x_test, y_test)
# print(results)

# plot_model(model, to_file='model.png')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# net = torch.load('./entire_torch.pth')
# net.to(device)
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # x_train = x_train / 255.0
# x_train = np.reshape(x_train, (60000, 1, 28, 28)).astype('float32') / 255.0
# x_test = np.reshape(x_test, (10000, 1, 28, 28)).astype('float32') / 255.0
# y_train = np.array(y_train).astype('int64')
# y_test = np.array(y_test).astype('int64')
#
# train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
# trainloader = data_utils.DataLoader(train, batch_size=50, shuffle=True)
#
# test = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
# testloader = data_utils.DataLoader(test, batch_size=1, shuffle=False)
#
# classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
#
# for epoch in range(2000):
#     running_loss = 0.0
#     exp_lr_scheduler.step()
#
#     for i, data in enumerate(trainloader):
#         inputs, labels = data[0].to(device), data[1].to(device)
#
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i == 0:
#             print("epoch {}, train_loss: {:.5f}".format(epoch, running_loss))
#     if epoch % 10 == 0:
#         # acc_val(net, testloader)
#         acc_val_classes(net, testloader, classes)
#
# input = np.reshape(x_train[1], (1,1,28,28)).astype('float32')
#
# a.eval()
# print(torch.argmax(a(torch.from_numpy(input))))
# plt.imshow(x_train[2])
# plt.show()