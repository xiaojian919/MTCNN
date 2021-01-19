import os
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class Sampling(Dataset):
    def __init__(self,data_path):
        '''

        :param data_path:某一尺寸的文件夹路径 例：r".\CelebA\12"
        '''
        self.datasets = []
        class_names = os.listdir(data_path)
        # class_names = ['negative', 'negative.txt', 'part', 'part.txt', 'positive', 'positive.txt']
        for a in range(3):
            #获取索引135的txt文件 通过txt找对应的图片
            for i,line in enumerate(open(os.path.join(data_path, class_names[2*a+1]), "r")):
                strs = line.split()
                img_filename = strs[0].strip()
                img_path = os.path.join(data_path, img_filename)
                # img = Image.open(image_file)
                self.datasets.append([img_path, strs[1:6]])
                # ['.\CelebA\12\part\0.jpg', ['2', '0.19618528610354224', '-0.0653950953678474', '-0.1880108991825613', '-0.2125340599455041']]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        data = self.datasets[item]
        self.img = Image.open(data[0])
        data_tensor = self.trans(Image.open(data[0]))
        category = np.array(int(data[1][0].strip())).reshape([-1])
        offset = []
        for i in range(1,5):
            offset.append(float(data[1][i].strip()))
        offset = np.stack(offset)
        return data_tensor,category,offset

    def trans(self, x):
        return transforms.Compose([
            # ToTensor()这个方法把图片0到255压缩到0到1
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])(x)


if __name__ == '__main__':
    # class_names = os.listdir(r".\CelebA\12")
    # print(class_names)
    # #['negative', 'negative.txt', 'part', 'part.txt', 'positive', 'positive.txt']
    # negative_txt = os.path.join(r".\CelebA\12",class_names[3])
    # negative_anno_file = open(negative_txt, "r")
    # for i, line in enumerate(negative_anno_file):
    #     strs = line.split()
    #     print(strs[1:6])
    #     image_filename = strs[0].strip()
    #     print(image_filename)
    #     image_file = os.path.join(r".\CelebA\12", image_filename)
    #     print(image_file)
    #     exit()
    #     img =Image.open(image_file)
    #     img.show()
    #     if i >1:
    #         exit()
    data_path = r".\CelebA\12"
    data = Sampling(data_path)
    dataloader = DataLoader(data, 10, shuffle=True)
    for i, (data_tensor,category,offset) in enumerate(dataloader):
        print(data_tensor.shape)#torch.Size([10, 3, 12, 12])
        print(category.shape)#torch.Size([10])
        print(offset.shape)#torch.Size([10, 4])
        print(type(offset))#<class 'torch.Tensor'>
        print(category)
        print(offset)
        break




