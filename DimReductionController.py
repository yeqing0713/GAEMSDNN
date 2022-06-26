import ModelController
import DNNDaset
import torch
import dgl
import torch.nn.functional as F
import UDFS
import sparse_learning
import numpy

def DR_preprocess_main(DR_type, drugFeatureVectors, targetFeatureVectors, Y, device, batch_size):
    if DR_type == 'DNNAutoEncode':
        #drugFeatureVectors1, targetFeatureVectors1 = get_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors, Y, 300,  device, batch_size)
        drugFeatureVectors2, targetFeatureVectors2 = get_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors, Y, 500, device, batch_size)
        drugFeatureVectors2 = drugFeatureVectors2.to("cpu")
        targetFeatureVectors2 = targetFeatureVectors2.to("cpu")
        #drugFeatureVectors1 = drugFeatureVectors1.to("cpu")
        #targetFeatureVectors1 = targetFeatureVectors1.to("cpu")
        drugFeatureVectors = torch.cat((drugFeatureVectors, drugFeatureVectors2), 1)
        targetFeatureVectors = torch.cat((targetFeatureVectors, targetFeatureVectors2), 1)
    elif DR_type == 'GCNAutoEncode':
        #drugFeatureVectors1, targetFeatureVectors1 = get_GCN_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors, Y, 300,  device)
        drugFeatureVectors2, targetFeatureVectors2 = get_GCN_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors, Y, 500, device)
        drugFeatureVectors2 = drugFeatureVectors2.to("cpu")
        targetFeatureVectors2 = targetFeatureVectors2.to("cpu")
        #drugFeatureVectors1 = drugFeatureVectors1.to("cpu")
        #targetFeatureVectors1 = targetFeatureVectors1.to("cpu")
        drugFeatureVectors = torch.cat((drugFeatureVectors, drugFeatureVectors2), 1)
        targetFeatureVectors = torch.cat((targetFeatureVectors, targetFeatureVectors2), 1)
        #drugFeatureVectors = drugFeatureVectors2
        #targetFeatureVectors = targetFeatureVectors2

    elif DR_type == 'VGAEAutoEncode':
        #drugFeatureVectors1, targetFeatureVectors1 = get_VGAEModel_fea(drugFeatureVectors, targetFeatureVectors, Y, 300, device)
        drugFeatureVectors2, targetFeatureVectors2 = get_VGAEModel_fea(drugFeatureVectors, targetFeatureVectors, Y, 500, device)
        drugFeatureVectors2 = drugFeatureVectors2.to("cpu")
        targetFeatureVectors2 = targetFeatureVectors2.to("cpu")
        #drugFeatureVectors1 = drugFeatureVectors1.to("cpu")
        #targetFeatureVectors1 = targetFeatureVectors1.to("cpu")
        drugFeatureVectors = torch.cat((drugFeatureVectors, drugFeatureVectors2), 1)
        targetFeatureVectors = torch.cat((targetFeatureVectors, targetFeatureVectors2), 1)
        #drugFeatureVectors = drugFeatureVectors2
        #targetFeatureVectors = targetFeatureVectors2

    elif DR_type == 'UDFS':
        [drugFeatureVectors1, targetFeatureVectors1] = get_UDFS_fea(drugFeatureVectors, targetFeatureVectors, Y, 500, device)
        drugFeatureVectors = drugFeatureVectors1
        targetFeatureVectors = targetFeatureVectors1
    return drugFeatureVectors, targetFeatureVectors

def get_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors_T, Y, out_c,device, batch_size):
    aa = drugFeatureVectors.shape
    model = ModelController.create_model('AutoEncoder', aa[1], out_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_Dataset22 = DNNDaset.CDNNDataset(drugFeatureVectors, Y)
    trainloader22 = torch.utils.data.DataLoader(train_Dataset22, batch_size=batch_size, shuffle=True, drop_last=False)
    epochs = 50
    for epoch in range(1, epochs + 1):
        ModelController.train_model('AutoEncoder', model, optimizer, train_loader=trainloader22, device=device)
    drugFeatureVectors1 = ModelController.test_model('AutoEncoder', model, drugFeatureVectors, device)
    drugFeatureVectors1 = drugFeatureVectors1.to(device)

    aa = targetFeatureVectors_T.shape
    model = ModelController.create_model('AutoEncoder', aa[1], out_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_Dataset22 = DNNDaset.CDNNDataset(targetFeatureVectors_T, Y.t())
    trainloader22 = torch.utils.data.DataLoader(train_Dataset22, batch_size=batch_size, shuffle=True, drop_last=False)
    epochs = 50
    for epoch in range(1, epochs + 1):
        ModelController.train_model('AutoEncoder', model, optimizer, train_loader=trainloader22, device=device)
    targetFeatureVectors1 = ModelController.test_model('AutoEncoder', model, targetFeatureVectors_T, device)
    targetFeatureVectors1 = targetFeatureVectors1.to(device)
    return drugFeatureVectors1, targetFeatureVectors1

def get_GCN_AutoEncoder_fea(drugFeatureVectors, targetFeatureVectors_T, Y, out_c, device):
    aa = drugFeatureVectors.shape
    knn_g = dgl.knn_graph(drugFeatureVectors, 10)
    model = ModelController.create_model('GCN_AutoEncoder', in_feats=aa[1], n_hidden=out_c, activation=F.relu, device=device, g= knn_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    for epoch in range(1, epochs + 1):
        ModelController.train_model('GCN_AutoEncoder', model, optimizer, features=drugFeatureVectors, device=device)
    drugFeatureVectors1 = ModelController.test_model('GCN_AutoEncoder', model, drugFeatureVectors, device)
    drugFeatureVectors1 = drugFeatureVectors1.to(device)

    aa = targetFeatureVectors_T.shape
    knn_g = dgl.knn_graph(targetFeatureVectors_T, 10)
    model = ModelController.create_model('GCN_AutoEncoder', in_feats=aa[1], n_hidden=out_c, activation=F.relu, device=device, g= knn_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    for epoch in range(1, epochs + 1):
        ModelController.train_model('GCN_AutoEncoder', model, optimizer, features=targetFeatureVectors_T, device=device)
    targetFeatureVectors1 = ModelController.test_model('GCN_AutoEncoder', model, targetFeatureVectors_T, device)
    targetFeatureVectors1 = targetFeatureVectors1.to(device)
    return drugFeatureVectors1, targetFeatureVectors1

def get_VGAEModel_fea(drugFeatureVectors, targetFeatureVectors_T, Y, out_c, device):
    aa = drugFeatureVectors.shape
    knn_g = dgl.knn_graph(drugFeatureVectors, 10)
    model = ModelController.create_model('VGAEModel', in_feats=aa[1], n_hidden=out_c, activation=F.relu, device=device, g= knn_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    for epoch in range(1, epochs + 1):
        ModelController.train_model('VGAEModel', model, optimizer, features=drugFeatureVectors, device=device)
    drugFeatureVectors1 = ModelController.test_model('VGAEModel', model, drugFeatureVectors, device)
    drugFeatureVectors1 = drugFeatureVectors1.to(device)

    aa = targetFeatureVectors_T.shape
    knn_g = dgl.knn_graph(targetFeatureVectors_T, 10)
    model = ModelController.create_model('VGAEModel', in_feats=aa[1], n_hidden=out_c, activation=F.relu, device=device, g= knn_g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    for epoch in range(1, epochs + 1):
        ModelController.train_model('VGAEModel', model, optimizer, features=targetFeatureVectors_T, device=device)
    targetFeatureVectors1 = ModelController.test_model('VGAEModel', model, targetFeatureVectors_T, device)
    targetFeatureVectors1 = targetFeatureVectors1.to(device)
    return drugFeatureVectors1, targetFeatureVectors1

def get_UDFS_fea(drugFeatureVectors, targetFeatureVectors_T, Y, out_c, device):
    num_cluster = 20
    Weight = UDFS.udfs(drugFeatureVectors, gamma=0.1, n_clusters=num_cluster)
    idx = sparse_learning.feature_ranking(Weight)
    idx = numpy.ascontiguousarray(idx)
    drugFeatureVectors1 = drugFeatureVectors[:, idx[0:out_c]]

    drugFeatureVectors1 = drugFeatureVectors1.to(device)

    num_cluster = 20
    Weight = UDFS.udfs(targetFeatureVectors_T, gamma=0.1, n_clusters=num_cluster)
    idx = sparse_learning.feature_ranking(Weight)
    idx = numpy.ascontiguousarray(idx)
    targetFeatureVectors1 = targetFeatureVectors_T[:, idx[0:out_c]]
    targetFeatureVectors1 = targetFeatureVectors1.to(device)
    return drugFeatureVectors1, targetFeatureVectors1