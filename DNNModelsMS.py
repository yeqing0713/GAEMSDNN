import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import numpy
from dgl.nn.pytorch import GraphConv

class RBMGoogleNet_MS(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device):
        super(RBMGoogleNet_MS, self).__init__()

        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.fc6 = nn.Linear(n_hidden, n_classes)
        self.fc4 = nn.Linear(n_classes * k, n_classes)
        self.fc5 = nn.Linear(n_hidden * k, n_classes)
        self.fc7 = nn.Linear(n_hidden * k, n_hidden)
        self.device = device

    def forward(self, xs):
        outputss = []
        output_c = []
        for ii in range(len(xs)):
            x = xs[ii]
            x = x.to(self.device)
            outputs = []
            for i, layer in enumerate(self.layers):
                if i != 0:
                    x = self.dropout(x)
                if i < len(self.layers) - 1:
                    x = F.relu(layer(x))
                    outputs.append(x)
                if i == len(self.layers) - 1:
                    output = layer(x)
                    outputs.append(output)
            output_c.append(output)
            outputss.append(outputs)

        for ii in range(len(outputss)):
            outputs = outputss[ii]
            for jj in range(len(outputs)):
                if ii != 0:
                    outputss[0][jj] = torch.cat((outputss[0][jj], outputss[ii][jj]), 1)

        outputs = outputss[0]
        output_c1 = []
        for jj in range(len(outputs)):
            if jj != len(outputs) - 1:
                aa = outputss[0][jj]
                output = self.fc7(aa)
                output = self.fc6(output)
                output_c1.append(output)
            else:
                aa = outputss[0][jj]
                output = self.fc4(aa)
                output_c1.append(output)

        return output_c, output_c1

class GCN_3_MS(nn.Module):
    def __init__(self,  in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device, gs):
        super(GCN_3_MS, self).__init__()
        self.gs = gs
        self.device = device
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, 10, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers1.append(nn.Linear(n_hidden, n_hidden))
        self.layers1.append(nn.Linear(n_hidden, 10))

        self.fc4 = nn.Linear(20, n_classes)
        self.fc5 = nn.Linear(10, n_classes)
        self.fc6 = nn.Linear(n_hidden, n_classes)
        self.fc7 = nn.Linear(2 * k, 2)
        self.fc8 = nn.Linear(20 * k, 2)

    def forward(self, xs):
        outputs = []
        aaa =[]
        bbb = []
        for ii in range(len(xs)):
            features = xs[ii]
            features = features.to(self.device)
            g = self.gs[ii]
            g = g.to(self.device)
            h = features
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g, h)

            x = features
            for i, layer in enumerate(self.layers1):
                if i != 0:
                    x = self.dropout(x)
                x = F.relu(layer(x))

            h1 = self.fc5(h)
            x1 = self.fc5(x)
            x = torch.cat((h, x), 1)
            h = self.fc4(x)
            outputs.append(h1)
            outputs.append(x1)
            outputs.append(h)
            if ii == 0:
                aaa = h
                bbb = x
            else:
                aaa = torch.cat((aaa, h), 1)
                bbb = torch.cat((bbb, x), 1)

        x = self.fc7(aaa)
        outputs.append(x)
        x = self.fc8(bbb)
        outputs.append(x)
        return outputs

class RBM_n_MS(nn.Module):
    def __init__(self,  in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device):
        super(RBM_n_MS, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        self.dropout = nn.Dropout(p=dropout)

        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers1.append(nn.Linear(n_hidden, n_hidden))
        self.layers1.append(nn.Linear(n_hidden, 20))

        self.fc5 = nn.Linear(20, n_classes)
        self.fc6 = nn.Linear(n_hidden, n_classes)
        self.fc7 = nn.Linear(2 * k, 2)
        self.fc8 = nn.Linear(20 * k, 2)

    def forward(self, xs):
        outputs = []
        bbb = []
        ccc = []
        for ii in range(len(xs)):
            features = xs[ii]
            features = features.to(self.device)

            x = features
            for i, layer in enumerate(self.layers1):
                if i != 0:
                    x = self.dropout(x)
                x = F.relu(layer(x))

            x1 = self.fc5(x)
            outputs.append(x1)
            if ii == 0:
                bbb = x
                ccc = x1
            else:
                bbb = torch.cat((bbb, x), 1)
                ccc = torch.cat((ccc, x1), 1)

        x = self.fc7(ccc)
        outputs.append(x)
        x = self.fc8(bbb)
        outputs.append(x)
        return outputs

class RBM_n_MSU(nn.Module):
    def __init__(self,  in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device):
        super(RBM_n_MSU, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        self.dropout = nn.Dropout(p=dropout)

        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers1.append(nn.Linear(n_hidden, n_hidden))
        self.layers1.append(nn.Linear(n_hidden, 10))

        self.fc5 = nn.Linear(10, n_classes)
        self.fc6 = nn.Linear(n_hidden, n_classes)
        self.fc7 = nn.Linear(2 * k, 2)
        self.fc8 = nn.Linear(10 * k, 2)

    def forward(self, xs):
        outputs = []
        bbb = []
        ccc = []
        for ii in range(len(xs)):
            features = xs[ii]
            features = features.to(self.device)

            x = features
            for i, layer in enumerate(self.layers1):
                if i != 0:
                    x = self.dropout(x)
                x = F.relu(layer(x))

            x1 = self.fc5(x)
            outputs.append(x1)
            if ii == 0:
                bbb = x
                ccc = x1
            else:
                bbb = torch.cat((bbb, x), 1)
                ccc = torch.cat((ccc, x1), 1)

        #x = self.fc7(ccc)
        #outputs.append(x)
        #x = self.fc8(bbb)
        #outputs.append(x)
        return outputs

class RBM_n_MS_D(nn.Module):
    def __init__(self,  in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device):
        super(RBM_n_MS_D, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        self.dropout = nn.Dropout(p=dropout)

        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers1.append(nn.Linear(n_hidden, n_hidden))

        self.fc4 = nn.Linear(n_hidden, 10)
        self.fc5 = nn.Linear(10, n_classes)
        self.fc6 = nn.Linear(n_hidden, n_classes)
        self.fc7 = nn.Linear(2 * k, 2)
        self.fc8 = nn.Linear(10 * k, 2)

    def forward(self, xs):
        outputs = []
        outputs1 = []
        bbb = []
        ccc = []
        for ii in range(len(xs)):
            features = xs[ii]
            features = features.to(self.device)

            x = features
            for i, layer in enumerate(self.layers1):
                if i != 0:
                    x = self.dropout(x)
                x = F.relu(layer(x))

            x = self.fc4(x)
            outputs1.append(x)
            x1 = self.fc5(x)
            outputs.append(x1)
            if ii == 0:
                bbb = x
                ccc = x1
            else:
                bbb = torch.cat((bbb, x), 1)
                ccc = torch.cat((ccc, x1), 1)

        x = self.fc7(ccc)
        outputs.append(x)
        x = self.fc8(bbb)
        outputs.append(x)
        return outputs, outputs1

class RBM_n_MS_D2(nn.Module):
    def __init__(self,  in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device):
        super(RBM_n_MS_D2, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        self.dropout = nn.Dropout(p=dropout)

        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers1.append(nn.Linear(n_hidden, n_hidden))

        self.fc4 = nn.Linear(n_hidden, 10)
        self.fc5 = nn.Linear(10, n_classes)
        self.fc6 = nn.Linear(n_hidden, n_classes)
        self.fc7 = nn.Linear(2 * k, 2)
        self.fc8 = nn.Linear(10 * k, 2)

    def forward(self, xs):
        outputs = []
        outputs1 = []
        outputs2 = []
        bbb = []
        ccc = []
        for ii in range(len(xs)):
            features = xs[ii]
            features = features.to(self.device)

            x = features
            for i, layer in enumerate(self.layers1):
                if i != 0:
                    x = self.dropout(x)
                x = F.relu(layer(x))
                if i == 0:
                    outputs2.append(x)

            x = self.fc4(x)
            x = F.relu(x)
            outputs1.append(x)
            x1 = self.fc5(x)
            outputs.append(x1)
            if ii == 0:
                bbb = x
                ccc = x1
            else:
                bbb = torch.cat((bbb, x), 1)
                ccc = torch.cat((ccc, x1), 1)

        x = self.fc7(ccc)
        outputs.append(x)
        x = self.fc8(bbb)
        outputs.append(x)
        return outputs, outputs1, outputs2

def train_MS(model, optimizer, features, labels, idx_train, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    optimizer.zero_grad()
#    features = features.to(device)
    output_c, output_c1 = model(features)

    loss_train = 0
    for ii in range(len(output_c)):
        output = output_c[ii]
        loss_train = loss_train + criterion(output[idx_train], torch.squeeze(labels[idx_train]))

    for ii in range(len(output_c1)):
        output = output_c1[ii]
        loss_train = loss_train + criterion(output[idx_train], torch.squeeze(labels[idx_train]))

    loss_train.backward()
    optimizer.step()

def test_MS(model, features, labels, idx_test):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        output_c, output_c1 = model(features)

    test_probabilitys = []
    test_labels = []
    for ii in range(len(output_c1)):
        output = output_c1[ii]
        probability = nn.functional.softmax(output, dim=1)
        probability = probability[idx_test]
        test_label = labels[idx_test]
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    return test_probabilitys, test_labels



def train_MS1_DL(model, optimizer, features, train_loader, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()

    for batch_idx, (data, target, inxs) in enumerate(train_loader):
        #print(inxs)
        #print(data.shape)
        optimizer.zero_grad()
        feas = []
        for ii in range(len(features)):
            feature = features[ii]
            feature = feature[inxs, :]
            feature = feature.to(device)
            feas.append(feature)

        #features, labels = data.to(device), target.to(device)
        labels = target.to(device)
        outputs = model(feas)

        loss_train = 0
        for ii in range(len(outputs)):
            output = outputs[ii]
            loss_train = loss_train + criterion(output, torch.squeeze(labels))

        loss_train.backward()
        optimizer.step()

def test_MS1_DL(model, features, labels, idx_test, device):
    model.eval()
    # features = features.to(device)
    #features = features[idx_test, :]
    feas = []
    for ii in range(len(features)):
        feature = features[ii]
        feature = feature[idx_test, :]
        feature = feature.to(device)
        feas.append(feature)
    labels = labels[idx_test]
    #features = features.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(feas)

    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)):
        output = outputs[ii]
        probability = nn.functional.softmax(output, dim=1)
        probability = probability
        test_label = labels
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    return test_probabilitys, test_labels

def test_MS1_DL1(model, features, labels, device):
    model.eval()
    # features = features.to(device)
    #features = features[idx_test, :]
    feas = []
    for ii in range(len(features)):
        feature = features[ii]
        #feature = feature[idx_test, :]
        feature = feature.to(device)
        feas.append(feature)
    #labels = labels[idx_test]
    #features = features.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(feas)

    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)-2, len(outputs)-1):
        output = outputs[len(outputs)-2]
        probability = nn.functional.softmax(output, dim=1)
        probability = probability
        test_label = labels
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    return test_probabilitys, test_labels


def train_MS1(model, optimizer, features, labels, idx_train, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    optimizer.zero_grad()
    #features = features.to(device)
    outputs = model(features)

    loss_train = 0
    for ii in range(len(outputs)):
        output = outputs[ii]
        loss_train = loss_train + criterion(output[idx_train], torch.squeeze(labels[idx_train]))

    loss_train.backward()
    optimizer.step()

def test_MS1(model, features, labels, idx_test):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        outputs = model(features)

    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)):
        output = outputs[ii]
        probability = nn.functional.softmax(output, dim=1)
        probability = probability[idx_test]
        test_label = labels[idx_test]
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    return test_probabilitys, test_labels

def train_MS1_U(model, optimizer, features, labels, idx_train, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    optimizer.zero_grad()
    #features = features.to(device)
    outputs = model(features)

    loss_train = 0
    for ii in range(len(outputs)):
        output = outputs[ii]
        loss_train = loss_train + criterion(output[idx_train], torch.squeeze(labels[idx_train]))

    loss_train.backward()
    optimizer.step()

def test_MS1_U(model, features, labels, idx_test):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        outputs = model(features)

    test_probabilitys = []
    test_labels = []
    output_total = 0
    for ii in range(len(outputs)):
        output = outputs[ii]
        output_total = output_total + output
        probability = nn.functional.softmax(output, dim=1)
        probability = probability[idx_test]
        test_label = labels[idx_test]
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    output_total = output_total/len(outputs)
    probability = nn.functional.softmax(output_total, dim=1)
    probability = probability[idx_test]
    test_label = labels[idx_test]
    test_probabilitys.append(probability)
    test_labels.append(test_label)

    return test_probabilitys, test_labels

def train_MS2(model, optimizer, features, labels, idx_train, device):
    criterion = nn.CrossEntropyLoss().to(device)
    criterion1 = nn.MSELoss().to(device)
    model.train()
    optimizer.zero_grad()
    #features = features.to(device)
    outputs, outputs1 = model(features)

    loss_train = 0
    for ii in range(len(outputs)):
        output = outputs[ii]
        loss_train = loss_train + criterion(output[idx_train], torch.squeeze(labels[idx_train]))
    loss_train = loss_train/len(outputs)

    loss_train1 = 0
    for ii in range(len(outputs1)):
        for jj in range(len(outputs1)):
            output1 = outputs1[ii]
            output2 = outputs1[jj]
            loss_train1 = loss_train1 + criterion1(output1, output2)
    loss_train1 = loss_train1/(len(outputs1) * len(outputs1))

    loss_train = loss_train * 3 + loss_train1

    loss_train.backward()
    optimizer.step()

def test_MS2(model, features, labels, idx_test):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        outputs, outputs1 = model(features)

    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)):
        output = outputs[ii]
        probability = nn.functional.softmax(output, dim=1)
        probability = probability[idx_test]
        test_label = labels[idx_test]
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    return test_probabilitys, test_labels

def train_MS3(model, optimizer, features, labels, idx_train, device):
    criterion = nn.CrossEntropyLoss().to(device)
    criterion1 = nn.MSELoss().to(device)
    model.train()
    optimizer.zero_grad()
    #features = features.to(device)
    outputs, outputs1, outputs2 = model(features)

    loss_train = 0
    for ii in range(len(outputs)):
        output = outputs[ii]
        loss_train = loss_train + criterion(output[idx_train], torch.squeeze(labels[idx_train]))
    loss_train = loss_train/len(outputs)

    loss_train1 = 0
    for ii in range(len(outputs1)):
        for jj in range(len(outputs1)):
            output1 = outputs1[ii]
            output2 = outputs1[jj]
            loss_train1 = loss_train1 + criterion1(output1, output2)
    loss_train1 = loss_train1/(len(outputs1) * len(outputs1))

    loss_train2 = 0
    for ii in range(len(outputs2)):
        for jj in range(len(outputs2)):
            output1 = outputs2[ii]
            output2 = outputs2[jj]
            loss_train2 = loss_train2 + criterion1(output1, output2)
    loss_train2 = loss_train2/(len(outputs2) * len(outputs2))

    loss_train = loss_train * 3 + loss_train1 + loss_train2

    loss_train.backward()
    optimizer.step()

def test_MS3(model, features, labels, idx_test):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        outputs, outputs1, outputs2 = model(features)

    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)):
        output = outputs[ii]
        probability = nn.functional.softmax(output, dim=1)
        probability = probability[idx_test]
        test_label = labels[idx_test]
        test_probabilitys.append(probability)
        test_labels.append(test_label)

    return test_probabilitys, test_labels