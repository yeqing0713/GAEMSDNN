# utils.py
#
# This is a file from GraphDTA.
#
# Modified by: Shugang Zhang
# Created: Wednesday, Aug 4th, 2021
# Last update: Thursday, Aug 5th, 2021

import os
from math import sqrt
from scipy import stats

import torch
from rdkit import Chem

import networkx as nx
import numpy as np
import random
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    #print(c_size)
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()

    edge_index1 = []
    for e1, e2 in g.edges:
        edge_index1.append([e1, e2])

    edge_index0 = []
    inx = random.randint(0, len(g.edges)-1)
    curr_inx = 0
    start_inx = 0
    end_inx = 0
    for e1, e2 in g.edges:
        if curr_inx == inx:
            start_inx = e1
            end_inx = e2
            break
        curr_inx = curr_inx + 1
    for e1, e2 in g.edges:
        if e1 == start_inx and e2 == end_inx:
            continue
        if e1 == end_inx and e2 == start_inx:
            continue
        edge_index0.append([e1, e2])

    edge_index2 = []
    start_inx = random.randint(0, c_size-1)
    end_inx = random.randint(0, c_size-1)
    while(1):
        if start_inx != end_inx:
            break
        end_inx = random.randint(0, c_size - 1)
    flag = 1
    flag1 = 1
    for e1, e2 in g.edges:
        if (e1 > start_inx and flag == 1):
            edge_index2.append([start_inx, end_inx])
            flag = 0

        if (e1 > end_inx and flag1 == 1):
            edge_index2.append([end_inx, start_inx])
            flag1 = 0
        edge_index2.append([e1, e2])
    return c_size, features, edge_index0, edge_index1, edge_index2

def get_GCNDATA(string):
    catgory = random.randint(0, 2)
    [c_size, features, edge_index0, edge_index1, edge_index2] = smile_to_graph(string)
    if (catgory == 0):
        GCNData = DATA.Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index0).transpose(1, 0),
                        y=torch.LongTensor(0))
    elif (catgory == 1):
        GCNData = DATA.Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index1).transpose(1, 0),
                        y=torch.LongTensor(1))
    else:
        GCNData = DATA.Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
                        y=torch.LongTensor(2))
    return GCNData, catgory, c_size


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        print(self.processed_paths[0])
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd):
        #assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print(i)
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]

            GCNData, catgory, c_size = get_GCNDATA(smiles)
            # convert SMILES to molecular representation using rdkit
            GCNData.target = torch.LongTensor([catgory])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0,1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci