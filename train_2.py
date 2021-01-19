import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from sampling import Sampling
from sklearn.metrics import r2_score
import nets_2


class Trainer:
    def __init__(self, net, save_path, dataset_path,valdataset_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
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

    def trainer(self,round_limit = 10):
        '''

        :param round_limit: 多少轮不更新，就停止训练
        :return:
        '''
        faceDataset = Sampling(self.dataset_path)
        valDataset = Sampling(self.valdataset_path)
        #batch_size一般不要超过百分之一 经验值
        dataloader = DataLoader(faceDataset, batch_size=400, shuffle=True, num_workers=4)
        valdataloader = DataLoader(valDataset, batch_size=100, shuffle=True, num_workers=4)
        category_r2_end = 0
        offset_r2_end = 0.8651080402047384
        epoch = 0
        round = 0
        self.net.train()
        while True:
            label_category = []
            label_offset = []
            out_category = []
            out_offset = []
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
                print("epoch:", epoch, "loss:", loss, " cls_loss:", cls_loss, " offset_loss", offset_loss)

            for i, (img_data_, category_, offset_) in enumerate(valdataloader):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device, dtype=torch.float32)
                offset_ = offset_.to(self.device, dtype=torch.float32)

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                output_offset = _output_offset.view(-1, 4)

                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                # cls_loss = self.cls_loss_fn(torch.sigmoid(output_category), category)

                offset_mask = torch.gt(category_, 0)
                offset = torch.masked_select(offset_, offset_mask)
                output_offset = torch.masked_select(output_offset, offset_mask)
                # offset_loss = self.offset_loss_fn(output_offset, offset)

                # 将普通的计算放到CPU上，以节省GPU资源
                label_category.extend(category.data.cpu().numpy())
                label_offset.extend(offset.data.cpu().numpy())
                out_category.extend(torch.sigmoid(output_category).data.cpu().numpy())
                out_offset.extend(output_offset.data.cpu().numpy())
                #下面两行代码只是对最后一个批次做了r2 并不是整体 所以不准确
                # category_r2 = r2_score(category.data.cpu().numpy(),torch.sigmoid(output_category).data.cpu().numpy())
                # offset_r2 = r2_score(offset.data.cpu().numpy(), output_offset.data.cpu().numpy())
                category_r2 = r2_score(label_category, out_category)
                offset_r2 = r2_score(label_offset,out_offset)

                print("epoch:", epoch,"category_r2:", category_r2,"offset_r2", offset_r2)
            epoch += 1

            if offset_r2 > offset_r2_end:
                torch.save(self.net.state_dict(), self.save_path)
                category_r2_end = category_r2
                offset_r2_end = offset_r2
                print("save success，offset_r2更新为{}".format(offset_r2_end))
                round = 0
            else:
                round += 1
                print("参数未更新,offset_r2仍为{},第{}次未更新".format(offset_r2_end,round))
                if round >= round_limit:
                    print("最终category_r2为{}，offset_r2为{}".format(category_r2_end,offset_r2_end))
                    break




if __name__ == '__main__':
    net = nets_2.Pnet()
    if not os.path.exists("./test"):
        os.makedirs("./test")
    trainer =Trainer(net, r'.\test\p_net.pth', r".\CelebA\12",r".\CelebAVal\12")
    trainer.trainer(10)

