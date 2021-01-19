import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from sampling import Sampling


class Trainer:
    def __init__(self, net, save_path, dataset_path,valdataset_path = None):
        '''

        :param net: 要训练的网络
        :param save_path: 存储路径
        :param dataset_path: 数据集路径
        :param valdataset_path:
        '''
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("使用cuda")
        else:
            self.device = torch.device("cpu")
        #形参变实参
        self.net = net.to(self.device)
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.valdataset_path = valdataset_path
        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
        else:
            print("NO Param")

    def trainer(self,stop_value):
        faceDataset = Sampling(self.dataset_path)
        #batch_size一般不要超过百分之一 经验值
        dataloader = DataLoader(faceDataset, batch_size=500, shuffle=True, num_workers=4)
        loss = 0
        epoch = 0
        self.net.train()
        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device,dtype = torch.float32)
                offset_ = offset_.to(self.device,dtype = torch.float32)

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                output_offset = _output_offset.view(-1, 4)

                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                # print("output_category:{}".format(output_category.shape))
                # print("category:{}".format(category.shape))
                cls_loss = self.cls_loss_fn(torch.sigmoid(output_category), category)


                offset_mask = torch.gt(category_, 0)
                offset = torch.masked_select(offset_,offset_mask)
                output_offset = torch.masked_select(output_offset,offset_mask)
                # print("output_offset:{}".format(output_offset.shape))
                # print("offset:{}".format(offset.shape))
                offset_loss = self.offset_loss_fn(output_offset, offset)

                loss = cls_loss + offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                print("epoch:",epoch,"loss:", loss, " cls_loss:", cls_loss, " offset_loss",offset_loss)
            epoch += 1
            torch.save(self.net.state_dict(), self.save_path)
            print("save success")

            if loss < stop_value:
                break

