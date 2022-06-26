import sklearn
from torch.nn import Linear, LayerNorm, ReLU
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch import TAGConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import SGConv
from dgl.nn.pytorch import GatedGraphConv
from dgl.nn.pytorch import GINConv
from dgl.nn.pytorch import GMMConv
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch import DotGatConv
import dgl
import torch.nn.functional as F
import numpy
from preprocess import mask_test_edges, mask_test_edges_dgl, sparse_to_tuple, preprocess_graph

class GCN_AutoEncoder(nn.Module):
    def __init__(self, in_feats, n_hidden, activation, device, g):
        super(GCN_AutoEncoder, self).__init__()
        self.g = g.to(device)
        self.gcn1 = GraphConv(in_feats, 800, activation=activation)
        self.gcn2 = GraphConv(800, n_hidden)

        self.gcn3 = GraphConv(n_hidden, 800, activation=activation)
        self.gcn4 = GraphConv(800, in_feats)
        self.nns = nn.Sigmoid()  # compress to a range (0, 1)

    def forward(self, x):
        x = self.gcn1(self.g, x)
        x = self.gcn2(self.g, x)
        encoded = self.nns(x)
        x = self.gcn3(self.g, x)
        x = self.gcn4(self.g, x)
        x = self.nns(x)
        decoded = x
        return encoded, decoded

class VGAEModel(nn.Module):
    def __init__(self, in_dim, n_hidden, activation, device, g):
        super(VGAEModel, self).__init__()
        self.device = device
        self.g = g.to(device)
        self.in_dim = in_dim
        self.hidden1_dim = 800
        self.hidden2_dim = n_hidden
        self.adj = self.g.adjacency_matrix().to_dense().to(device)
        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]

        self.weight_tensor, self.norm = self.compute_loss_para(self.adj)

        self.layers = nn.ModuleList(layers)

    def encoder(self, features):
        h = self.layers[0](self.g, features)
        self.mean = self.layers[1](self.g, h)
        self.log_std = self.layers[2](self.g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, features):
        z = self.encoder(features)
        adj_rec = self.decoder(z)
        return z, adj_rec

    def get_loss(self, logits):
        #weight_tensor, norm = self.compute_loss_para(self.adj)
        loss = self.norm * F.binary_cross_entropy(logits.view(-1), self.adj.view(-1), weight=self.weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * self.log_std - self.mean ** 2 - torch.exp(self.log_std) ** 2).sum(
            1).mean()
        loss -= kl_divergence
        return loss

    def compute_loss_para(self, adj):
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        weight_mask = adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(self.device)
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor, norm

def train_VGAEModel(model, optimizer, data, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    encoded, decoded = model(data)
    loss = model.get_loss(decoded)
    loss.backward()
    optimizer.step()

def test_VGAEModel(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        encoded, decoded_data = model(data)
    return encoded

def train_GCN_AutoEncoder(model, optimizer, data, device):
    model.train()
    data = data.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer.zero_grad()
    encoded, decoded = model(data)
    loss = criterion(decoded, data)
    loss.backward()
    optimizer.step()

def test_GCN_AutoEncoder(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        encoded, decoded_data = model(data)
    return encoded