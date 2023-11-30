import io
import re

from flask import Blueprint, jsonify  # 从flask导入blueprint模块
from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from matplotlib.figure import Figure #在flask中最好直接使用matplotlib而不是pyplot
import base64

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .forms import Test_Dataset_Path

#设置蓝图的相关信息
bp=Blueprint("front",__name__)#创建蓝图对象，必须指定两个参数。bp是蓝图的名称，__name__表示蓝图所在模块
#前台访问不需要前缀（首页）

# #定义超参数
# n_epochs = 3 #epoch的数量定义了我们将循环整个训练数据集的次数
# batch_size_train = 64
batch_size_test = 1000
# learning_rate = 0.01 #稍后将使用的优化器的超参数
# momentum = 0.5 #稍后将使用的优化器的超参数
# log_interval = 10
# random_seed = 1
# torch.manual_seed(random_seed)
#
# # 设置数据集下载路径
# dataset_path = './data'  # 将数据集保存在Flask应用的"data"目录下
#
# #下载数据集
# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST(dataset_path, train=True, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=batch_size_train, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST(dataset_path, train=False, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=batch_size_test, shuffle=True)
#
# examples = enumerate(test_loader) #创建了一个test_loader的enumerate对象
# batch_idx, (example_data, example_targets) = next(examples)
# # print(example_targets) #打印图片实际对应的数字标签
# # print(example_data.shape) #一批测试数据是一个形状张量 1000个例子的28*28灰度（没有rgb）
#
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# # plt.show()
#
class Net(nn.Module):
    #定义模型的结构，包括两个卷积层conv1和conv2，一个二维dropout层conv2_drop，和两个全连接层fc1和fc2
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    #定义了数据在模型中前向传播的流程。通过卷积层和池化层，数据被压缩和提取特征，最后通过全连接层得到分类结果
    #F.relu是激活函数，F.max_pool2d是最大池化层，F.dropout是dropout操作，F.log_softmax是对最终输出进行softmax操作并取对数，通常在多分类问题中作为输出层使用
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# #初始化网络和优化器
# network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                       momentum=momentum)
#
# train_losses = []
# train_counter = []
# test_losses = []
# test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
#
# def train(epoch):
#     network.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = network(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
#                                                                            len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader),
#                                                                            loss.item()))
#             train_losses.append(loss.item())
#             train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
#             torch.save(network.state_dict(), './model.pth')
#             torch.save(optimizer.state_dict(), './optimizer.pth')
#
# def test():
#     network.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = network(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()
#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
#     print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#
# train(1)
#
# test()  # 不加这个，后面画图就会报错：x and y must be the same size
# for epoch in range(1, n_epochs + 1):
#     train(epoch)
#     test()
#
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# # plt.show()
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# with torch.no_grad():
#     output = network(example_data)
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
# # ----------------------------------------------------------- #
#
# continued_network = Net()
# continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#
# network_state_dict = torch.load('model.pth')
# continued_network.load_state_dict(network_state_dict)
# optimizer_state_dict = torch.load('optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)
#
# # 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# # 不然报错：x and y must be the same size
# # 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
# for i in range(4, 9):
#     test_counter.append(i * len(train_loader.dataset))
#     train(i)
#     test()
#
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()

#加载模型
network=Net()
network.load_state_dict(torch.load('model.pth'))

@bp.route('/',methods=['POST','GET'])
def home():
    if request.method=='GET':
        return render_template('front/frontPage.html')
    else:
        # 定义路径，来加载测试数据
        form = Test_Dataset_Path(request.form)
        if form.validate():
            test_dataset_path = request.form.get('path')
        # print(test_dataset_path)
        # 获得上传的图片数据
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(test_dataset_path, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size_test, shuffle=True)
        # 处理图片数据
        examples = enumerate(test_loader)  # 创建了一个test_loader的enumerate对象
        batch_idx, (example_data, example_targets) = next(examples)
        print("example_data的规模是："+str(len(example_data)))
        # 输出图片
        fig = Figure()
        for i in range(6):
            # generate the figure without using pyplot
            # print("i是"+str(i))
            ax=fig.add_subplot(2, 3, i + 1)
            # print("i是" + str(i))
            # tight_layout()
            ax.imshow(example_data[i][0], cmap='gray', interpolation='none')
            ax.set_title("Ground Truth: {}".format(example_targets[i]))
            ax.set_xticks([])
            ax.set_yticks([])
        buf=io.BytesIO()
        fig.savefig(buf,format='png')
        imagedata=base64.b64encode(buf.getbuffer()).decode('ascii')
        # return f"<img src='data:image/png;base64,{imagedata}'/>"
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        # test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return render_template('front/predict.html',loss=test_loss,accuracy=100. * correct / len(test_loader.dataset),
                               imagedata=imagedata)

#定义API路由，接收上传的图像，处理并使用模型进行预测，并返回每个类的概率
@bp.route('/predict', methods=['POST','GET'])
def predict():
    return render_template('front/predict.html')