import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def create_model_instance_SL(dataset_type, model_type, n_parties, class_num=10):
    client_nets = {net_i: None for net_i in range(n_parties)}
    # return VGG9()
    # return EMNIST_CNN1(),EMNIST_CNN2()
    if dataset_type == 'CIFAR10':
        server = AlexNet_DF2()
        for net_i in range(n_parties):
            net = AlexNet_DF1()
            client_nets[net_i] = net
        return client_nets, server
    elif dataset_type == 'image100':
        return IMAGE100_VGG16_1(), IMAGE100_VGG16_2()
    elif dataset_type == 'UCIHAR':
        return CNN_HAR1(), CNN_HAR2()
    elif dataset_type == 'SPEECH':
        return M5_1(), M5_2()
    

def create_model_instance_SL_two_splits(dataset_type, model_type, n_parties, class_num=10):
    client_nets = {net_i: None for net_i in range(n_parties)}
    if dataset_type == 'CIFAR10':
        server = AlexNet_DF2b()
        for net_i in range(n_parties):
            net = (AlexNet_DF1a(), AlexNet_DF2c())
            client_nets[net_i] = net
        return client_nets, server

class AlexNet_DF1(nn.Module):
    def __init__(self):
        super(AlexNet_DF1, self).__init__()
        self.features = nn.Sequential(
            #layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 2
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 3
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #layer 4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class AlexNet_DF2(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet_DF2, self).__init__()
        
        # self.f2= nn.Sequential(
            
        # )

        self.classifier = nn.Sequential(
            #layer 6
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            #layer 7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            #layer 8
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        # x = self.f2(x)
        x = x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
class AlexNet_DF1a(nn.Module):
    def __init__(self):
        super(AlexNet_DF1, self).__init__()
        self.features = nn.Sequential(
            #layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 2
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 3
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #layer 4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class AlexNet_DF2b(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet_DF2b, self).__init__()
        self.features = nn.Sequential(
            #layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            #layer 6
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            #layer 7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet_DF2c(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet_DF2c, self).__init__()
        self.classifier = nn.Sequential(
            #layer 8
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class EMNIST_CNN1(nn.Module):
    def __init__(self):
        super(EMNIST_CNN1,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(        
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        return out_conv2

class EMNIST_CNN2(nn.Module):
    def __init__(self):
        super(EMNIST_CNN2,self).__init__()
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self,out_conv2):
        output = out_conv2.view(-1,7*7*64)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

class IMAGE100_VGG16_1(nn.Module):
    def __init__(self, class_num=100):
        super(IMAGE100_VGG16_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),
            
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class IMAGE100_VGG16_2(nn.Module):
    def __init__(self, class_num=100):
        super(IMAGE100_VGG16_2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512*25, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = x.view(x.size(0), 512*25)
        x = self.classifier(x)
        return x


class CNN_HAR(nn.Module):
    def __init__(self):
        super(CNN_HAR, self).__init__()
        #定义第一个卷积层
        width = 1
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=int(12*width),          #输出高度12
                      kernel_size=3,            #卷积核尺寸3*3
                      stride=1,
                      padding=1),               #(1*128*9)-->(12*128*9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #(12*128*9)-->(12*64*4)
        )

        #定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(12*width),
                      out_channels=int(32*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),               #(12*64*4)-->(32*64*4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #池化后：(32*32*2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(32*width),
                      out_channels=int(64*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),                #(32*32*2)-->(64*32*2)
            nn.ReLU()
        )

        #定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(int(64*32*2*width),int(1024*width)),              #长方体变平面
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(int(1024*width),6)
        )

    #定义网络的前向传播路径
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],-1) #展平多维的卷积图层
        output = self.classifier(x)
        return output
    

class CNN_HAR(nn.Module):
    def __init__(self):
        super(CNN_HAR, self).__init__()
        #定义第一个卷积层
        width = 1
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=int(12*width),          #输出高度12
                      kernel_size=3,            #卷积核尺寸3*3
                      stride=1,
                      padding=1),               #(1*128*9)-->(12*128*9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #(12*128*9)-->(12*64*4)
        )

        #定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(12*width),
                      out_channels=int(32*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),               #(12*64*4)-->(32*64*4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #池化后：(32*32*2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(32*width),
                      out_channels=int(64*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),                #(32*32*2)-->(64*32*2)
            nn.ReLU()
        )

        #定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(int(64*32*2*width),int(1024*width)),              #长方体变平面
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(int(1024*width),6)
        )

    #定义网络的前向传播路径
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],-1) #展平多维的卷积图层
        output = self.classifier(x)
        return output


class CNN_HAR1(nn.Module):
    def __init__(self):
        super(CNN_HAR1, self).__init__()
        
        width = 1
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=int(12*width),          #输出高度12
                      kernel_size=3,            #卷积核尺寸3*3
                      stride=1,
                      padding=1),               #(1*128*9)-->(12*128*9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #(12*128*9)-->(12*64*4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(12*width),
                      out_channels=int(32*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),               #(12*64*4)-->(32*64*4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # (32*32*2)
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CNN_HAR2(nn.Module):
    def __init__(self):
        super(CNN_HAR2, self).__init__()
        width = 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(32*width),
                      out_channels=int(64*width),
                      kernel_size=3,
                      stride=1,
                      padding=1),                #(32*32*2)-->(64*32*2)
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(64*32*2*width), int(1024*width)), 
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(int(1024*width),6)
        )

    def forward(self,x):
        x = self.conv3(x)
        x = x.view(x.shape[0],-1) 
        output = self.classifier(x)
        return output


class M5_1(nn.Module):
    def __init__(self, n_input=1, stride=16, n_channel=32):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            # nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
            # nn.BatchNorm1d(2 * n_channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(4),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # x = F.avg_pool1d(x, x.shape[-1])
        return x


class M5_2(nn.Module):
    def __init__(self, n_output=35, n_channel=32):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            # nn.BatchNorm1d(2 * n_channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool1d(4),

            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        self.fc_layer = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):

        x = self.conv_layer(x)
        x = F.avg_pool1d(x, x.shape[-1])

        x = x.permute(0, 2, 1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=2).squeeze()