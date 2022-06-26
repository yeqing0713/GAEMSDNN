import DNNDaset
import torch
import ModelController
import dgl
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
import numpy as np

def create_graph(graph_type, train_fea, train_label, fea, label):
    if graph_type == 'knn_graph':
        return Graph_create_by_DNN(train_fea, train_label, fea)
    elif graph_type == 'gdc_knn_graph':
        return Graph_create_by_DNN_and_GDC(train_fea, train_label, fea, label)

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def Graph_create_by_DNN(train_fea, train_label, fea):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 10000
    train_Dataset = DNNDaset.CDNNDataset(train_fea, train_label)
    trainloader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    aa = train_fea.shape
    model = ModelController.create_model('RBMNet_3_GCN', aa[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(1, epochs + 1):
        ModelController.train_model('RBMNet_3_GCN', model, optimizer, train_loader=trainloader, device=device)
    fea = ModelController.test_model('RBMNet_3_GCN', model, fea, device)

    fea = fea.to("cpu")
    knn_g = dgl.knn_graph(fea, 10)

    return knn_g

def Graph_create_by_label(train_fea, train_label, fea, label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    aa = label.shape
    dist2 = torch.zeros((aa[0], aa[0]))
    for ii in range(0, aa[0]):
        for jj in range(0, aa[0]):
            if label[ii] == 1 and label[jj] == 1:
                dist2[ii][jj] = 1
    return dist2

def adj2edgeindex(adj):
    aa = adj.shape
    inx_DTI = torch.where(adj > 0)
    aa = inx_DTI[0]
    bb = inx_DTI[1]
    aa = torch.as_tensor(aa, dtype=torch.long)
    bb = torch.as_tensor(bb, dtype=torch.long)

    edge_index = torch.tensor([aa.numpy(), bb.numpy()], dtype=torch.long)
    edge_attr = adj[aa.numpy(), bb.numpy()]
    return edge_index, edge_attr

def Graph_create_by_DNN_and_GDC(train_fea, train_label, features, labels):
    knn_g = Graph_create_by_DNN(train_fea, train_label, features)
    cc = knn_g.num_edges()
    aa = knn_g.edges()[0]
    aa= torch.as_tensor(aa, dtype=torch.long)
    aa = np.array(aa)
    bb = knn_g.edges()[1]
    bb = torch.as_tensor(bb, dtype=torch.long)
    bb = np.array(bb)
    edge_index = torch.tensor([aa, bb], dtype=torch.long)
    edge_attr = torch.from_numpy(np.zeros((cc, 1)) + 1)
    edge_attr = torch.as_tensor(edge_attr, dtype=torch.float)

    edge_attr = edge_attr.squeeze(1)
    pyGdata = Data(x=features, edge_index=edge_index, y=labels, edge_attr=edge_attr)
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    pyGdata = gdc(pyGdata)
    edge_index = pyGdata.edge_index

    aa = torch.as_tensor(edge_index[0], dtype=torch.long)
    aa = np.array(aa)

    bb = torch.as_tensor(edge_index[1], dtype=torch.long)
    bb = np.array(bb)

    edge_attr = pyGdata.edge_attr
    g = dgl.graph((aa, bb))
    g.edata['w'] = edge_attr
    return g

def pygGraph_create_by_DNN(train_fea, train_label, features, labels):
    knn_g = Graph_create_by_DNN(train_fea, train_label, features)
    cc = knn_g.num_edges()
    num_nodes = knn_g.num_nodes()
    aa = knn_g.edges()[0]
    aa= torch.as_tensor(aa, dtype=torch.long)
    aa = np.array(aa)
    bb = knn_g.edges()[1]
    bb = torch.as_tensor(bb, dtype=torch.long)
    bb = np.array(bb)
    edge_index = torch.tensor([aa, bb], dtype=torch.long)
    edge_attr = torch.from_numpy(np.zeros((cc, 1)) + 1)
    edge_attr = torch.as_tensor(edge_attr, dtype=torch.float)

    edge_attr = edge_attr.squeeze(1)
    pyGdata = Data(x=features, edge_index=edge_index, y=labels, edge_attr=edge_attr)
    return pyGdata
'''
def parper_data_for_torch_geometric(x, y, adj):
    edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
    print(edge_index.shape)
    print(edge_attr.shape)
    #edge_index, edge_attr = adj2edgeindex(adj)
    #data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
    return edge_index, edge_attr
'''

