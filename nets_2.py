import torch
import torch.nn as nn

class Pnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(# N 3 12 12
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d((2,2),2)
        )# N 10 5 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,stride=1),# N 16 3 3
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),#N 32 1 1
            nn.PReLU()
        )
        self.conv4category = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)#N 1 1 1
        self.conv4offset = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)  # N 4 1 1

    def forward(self,x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        category = self.conv4category(y3)
        offset = self.conv4offset(y3)
        return category,offset


class Rnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(# N 3 24 24
            nn.Conv2d(in_channels=3, out_channels=28,kernel_size=3,stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d((3,3),2)
        )#N 28 11 11
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d((3, 3), 2)
        )#N 48 4 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1, padding=0),#N 64 3 3
            nn.PReLU()
        )
        self.linear = nn.Linear(in_features=3*3*64,out_features=128)#N 128
        self.linearcategory = nn.Linear(in_features=128,out_features=1) # N 1
        self.linearoffset = nn.Linear(in_features=128, out_features=4)  # N 4

    def forward(self,x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        #输入全连接层要控制输入形状
        y3 = y3.view(y3.size(0), -1)
        y4 = self.linear(y3)
        category = self.linearcategory(y4)
        offset = self.linearoffset(y4)
        return category, offset


class Onet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(  # N 3 48 48
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d((3, 3), 2)
        )  # N 32 23 23

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d((3, 3), 2)
        )#N 64 10 10

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d((2, 2), 2)
        )#N 64 4 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),#N 128 3 3
            nn.PReLU()
        )
        self.linear = nn.Linear(in_features=3 * 3 * 128, out_features=256)  # N 256
        self.linearcategory = nn.Linear(in_features=256, out_features=1)  # N 1
        self.linearoffset = nn.Linear(in_features=256, out_features=4)  # N 4

    def forward(self,x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        #输入全连接层要控制输入形状
        y4 = y4.view(y4.size(0), -1)
        y5 = self.linear(y4)
        category = self.linearcategory(y5)
        offset = self.linearoffset(y5)
        return category, offset







if __name__ == '__main__':
    #p网络测试
    x1 = torch.randn((10,3,12,12))
    print(x1.shape)
    pnet = Pnet()
    pcat,poff = pnet(x1)
    print(pcat.shape)#torch.Size([10, 1, 1, 1])
    print(poff.shape)#torch.Size([10, 4, 1, 1])


    # r网络测试
    # x2 = torch.randn((10, 3, 24, 24))
    # print(x2.shape)
    # rnet = Rnet()
    # rcat, roff = rnet(x2)
    # print(rcat.shape)#torch.Size([10, 1])
    # print(roff.shape)#torch.Size([10, 4])

    # o网络测试
    # x3 = torch.randn((10, 3, 48, 48))
    # print(x3.shape)
    # onet = Onet()
    # ocat, ooff = onet(x3)
    # print(ocat.shape)#torch.Size([10, 1])
    # print(ooff.shape)#torch.Size([10, 4])
