import torch
from torch.utils.data import Dataset
from random import shuffle
import numpy as np

class CDNNDataset(Dataset):
    def __init__(self, fea, label):
        self.train_X = fea
        self.train_Y = label
        self.train_Y = torch.as_tensor(self.train_Y, dtype=torch.long)
#        self.train_Y = self.train_Y.squeeze(1)

    def __len__(self):
        return self.train_X.shape[0]

    def __getitem__(self, index):
        return self.train_X[index, :], self.train_Y[index]


class CDNNDatasetMS(Dataset):
    def __init__(self, fea, label):
        self.train_X = fea
        self.train_Y = label
        self.train_Y = torch.as_tensor(self.train_Y, dtype=torch.long)
#        self.train_Y = self.train_Y.squeeze(1)

    def __len__(self):
        return self.train_X.shape[0]

    def __getitem__(self, index):
        return self.train_X[index, :], self.train_Y[index], index


class CDNNDataset1(Dataset):
    def __init__(self, drug_fea, target_fea, Y, ratio):
        self.drug_fea = drug_fea
        self.target_fea = target_fea
        Y = Y
        num_DTI = len(Y[Y == 1])
        a = Y.shape
        num_total = a[1] * a[0]

        num_unDTI = num_total - num_DTI
        inx_DTI = np.where(Y == 1)
        inx_unDTI = np.where(Y == 0)

        inx1 = [i for i in range(num_unDTI)]
        shuffle(inx1)
        inx_unDTIused0 = inx_unDTI[0][inx1[0:num_DTI * ratio]]
        inx_unDTIused1 = inx_unDTI[1][inx1[0:num_DTI * ratio]]
        self.inxs0 = np.concatenate((inx_DTI[0], inx_unDTIused0))
        self.inxs1 = np.concatenate((inx_DTI[1], inx_unDTIused1))
        pos_Y = torch.zeros((num_DTI, 1)) + 1
        neg_Y = torch.zeros((num_DTI * ratio, 1))
        self.label = torch.cat((pos_Y, neg_Y))
        self.label = torch.as_tensor(self.label, dtype=torch.long)
        print('self.inxs0')
        print(self.inxs0.shape)
        print('label')
        print(self.label.shape)
        self.label = self.label.squeeze(1)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        drug_fea_i = self.drug_fea[self.inxs0[index], :]
        target_fea_i = self.target_fea[self.inxs1[index], :]
        X = torch.cat((drug_fea_i, target_fea_i), 0)
        return X, self.label[index]

from torch.utils.data import DataLoader

