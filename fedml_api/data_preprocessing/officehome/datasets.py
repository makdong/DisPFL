import logging
import pdb
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR100, CIFAR10
import deeplake
import torch

class OFFICEHOME_truncated(data.Dataset):

    def __init__(self, root, cache_data=None, cache_target=None, dataidxs=None, train=True, transform=None, target_transform=None, download=True):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__(cache_data, cache_target)

    def __build_truncated_dataset__(self,cache_data, cache_target):
        if cache_data == None:
            # ds = deeplake.load('hub://activeloop/office-home-domain-adaptation')
            # officehome_dataobj = ds.pytorch(num_workers = 2, batch_size= 15500, transform = {'images': self.transform, 'domain_categories': None}, shuffle = False, drop_last = True)
            # cache_list = list(iter(officehome_dataobj))
            # [(X of batch1, y of batch1), (X of batch2, y of batch2), ..., (X of batch_last, y of batch_last)]
            # #_batch * 3 * 50 * 50 ==> batch size * RGB * image size
            # to get whole data, set the batch size as number of images, and set the drop_last as true.
            
            # train = true / false 보고 구분
            if(self.train == True):
                data = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/X_train.pt')
                target = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/y_train.pt')
            else:
                data = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/X_test.pt')
                target = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/y_test.pt')
            # domain 안에서 shuffle

            # dataloder = DataLoader(dataset, batch_size=128)
        else:
            
            # officehome_dataobj = cache_data_set
            # print(officehome_dataobj)
            # data = officehome_dataobj.data # 15500 * 3 * 64 * 64
            # target = np.array(officehome_dataobj.targets)
            data = cache_data.numpy()
            target = cache_target.numpy()
            
        if self.dataidxs is not None:
            # 전체 data중 idx 에 해당하는거 갖고오기
            data = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/X.pt')
            target = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/y.pt')

            data = data[self.dataidxs]
            target = target.flatten()[1::2][self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
