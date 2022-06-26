import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import numpy
from tqdm import tqdm
from torch_geometric.nn import SAGPooling, GCNConv, SAGEConv, global_mean_pool as gap, global_max_pool as gmp
import torchvision.models

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 800),
            nn.Tanh(),
            nn.Linear(800, out_channels),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, 800),
            nn.Tanh(),
            nn.Linear(800, in_channels),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )
        self.nns = nn.Sigmoid()  # compress to a range (0, 1)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        encoded = self.nns(encoded)
        return encoded, decoded

def train_AutoEncoder(model, optimizer, train_loader, device):
    model.train()
    criterion = nn.MSELoss().to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        encoded, decoded = model(data)
        loss = criterion(decoded, data)
        loss.backward()
        optimizer.step()

def test_AutoEncoder(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        encoded, decoded_data = model(data)
    return encoded


class RBMTansDTI(nn.Module):
    def __init__(self, t_in_fea):
        super(RBMTansDTI, self).__init__()
        embedding_size = 768
        num_filters = 768
        self.max_seq_len = 128
        self.tconv1 = nn.Conv1d(in_channels=embedding_size,  out_channels=num_filters,  kernel_size=8)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.tconv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.tconv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters * 3, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(num_filters * 3)
        self.tmaxp = nn.MaxPool1d(kernel_size=self.max_seq_len - 3*7)
        self.m = torch.nn.AdaptiveAvgPool2d((64, 768))

        self.fcD = nn.Linear(num_filters * 3, num_filters * 3)
        self.fcT = nn.Linear(t_in_fea, num_filters * 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(num_filters * 3 * 2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)
        self.d_frames = self.max_seq_len
        self.d_dims = embedding_size
        self.t_in_fea = t_in_fea

    def forward(self, fea):
        #inputs = self.tokenizer(xd, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        #print(xd)
        #print(fea.shape)
        drug_dim = self.d_frames * self.d_dims
        xd = fea[:, 0:drug_dim]
        xd = xd.reshape(xd.shape[0], self.d_frames, self.d_dims)
        xt = fea[:, drug_dim:fea.shape[1]]
        #print(xd)
        #print(xd.shape)
        #print(xt.shape)
        #print(self.t_in_fea)

        outputs = xd.permute(0, 2, 1)
        #print(outputs.shape)
        #print(outputs.shape)
        embed_xd = F.relu(self.tconv1(outputs))
        #print(embed_xd.shape)
        embed_xd = self.bn1(embed_xd)
        embed_xd = F.relu(self.tconv2(embed_xd))
        embed_xd = self.bn2(embed_xd)
        #print(embed_xd.shape)
        embed_xd = F.relu(self.tconv3(embed_xd))
        embed_xd = self.bn3(embed_xd)
        embed_xd = self.tmaxp(embed_xd)
        embed_xd = embed_xd.view(-1, embed_xd.size(1))
        #print(embed_xd.shape)
        embed_xd = F.relu(self.fcD(embed_xd))
        embed_xt = F.relu(self.fcT(xt))

        x = torch.cat([embed_xd, embed_xt], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        #outputs = []
        #outputs.append(x)
        return x

def train_RBMTansDTI(model, optimizer, device, train_loader):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    #features = features.to(device)

    lossss = 0
    for batch_idx, (fea, labels) in enumerate(train_loader):
        #print(target_feas)
        optimizer.zero_grad()
        #Input_ids = Input_ids.to(device)
        #Token_type_ids = Token_type_ids.to(device)
        #drug_feas['input_ids'] = drug_feas['input_ids'].to(device)
        #drug_feas['attention_mask'] = drug_feas['attention_mask'].to(device)
        #input_ids = input_ids.to(device)
        #attention_mask = attention_mask.to(device)
        fea = fea.to(device)
        labels = labels.to(device)
        print(fea)
        outputs = model(fea)
        print(outputs)
        loss_train = 0
        #for ii in range(len(outputs)):
        #    output = outputs[ii]
        loss_train = loss_train + criterion(outputs, torch.squeeze(labels))
        lossss = loss_train
        loss_train.backward()
        optimizer.step()
    #print(str(lossss.cpu().detach().numpy()), end = ' ')

def test_RBMTansDTI(model, device, labels, test_loader):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        ii = 0
        for batch_idx, (fea, labels) in enumerate(test_loader):
            #input_ids = input_ids.to(device)
            #attention_mask = attention_mask.to(device)
            #inputs = inputs.to(device)

            fea = fea.to(device)

            labels = labels.to(device)
            outputs = model(fea)

            if ii == 0:
                outputss = outputs
                labelss = labels
                ii = 1
            else:
                outputss = torch.cat((outputss, outputs), 0)
                labelss = torch.cat((labelss, labels), 0)

    outputs=[]
    outputs.append(outputss)
    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)):
        output = outputs[ii]

        probability = nn.functional.softmax(output, dim=1)
        probability = probability.cpu()
        labelss = labelss.cpu()

        test_probabilitys.append(probability)
        test_labels.append(labelss)

    return test_probabilitys, test_labels

class CNNTansDTI2D(nn.Module):
    def __init__(self, t_in_fea):
        super(CNNTansDTI2D, self).__init__()
        embedding_size = 768
        num_filters = 32
        self.max_seq_len = 128

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 5),
                stride=1,
                padding=(0, 2)
            ),                               #维度变换(1,28,28) --> (16,768,128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4))      #维度变换(16,28,28) --> (16,768,32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, 5),
                stride=1,
                padding=(0, 2)
            ),                               #维度变换(16,768,64) --> (16,768,64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4))      #维度变换(16,768,64) --> (16,768,8)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1)
            ),                               #维度变换(16,768,64) --> (16,768,64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))      #维度变换(16,768,64) --> (16,768,4)
        )

        self.fcD = nn.Linear(num_filters * 768 * 4, num_filters * 3)
        self.fcT = nn.Linear(t_in_fea, num_filters * 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(num_filters * 3 * 2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)
        self.d_frames = self.max_seq_len
        self.d_dims = embedding_size
        self.t_in_fea = t_in_fea
        self.m = torch.nn.AdaptiveAvgPool2d((128, 768))

    def forward(self, fea):
        drug_dim = self.d_frames * self.d_dims
        xd = fea[:, 0:drug_dim]
        xd = xd.reshape(xd.shape[0], self.d_frames, self.d_dims)
        #print(xd.shape)
        xd = self.m(xd)
        #print(xd.shape)
        outputs = xd.permute(0, 2, 1)
        outputs = outputs.unsqueeze(1)
        xt = fea[:, drug_dim:fea.shape[1]]
        #print(outputs.shape)

        embed_xd = self.conv1(outputs)
        #print(embed_xd.shape)
        embed_xd = self.conv2(embed_xd)
        #print(embed_xd.shape)
        embed_xd = self.conv3(embed_xd)

        #print(embed_xd.shape)
        embed_xd = embed_xd.view(-1, embed_xd.shape[1] * embed_xd.shape[2] * embed_xd.shape[3])
        #print(embed_xd.shape)
        embed_xd = F.relu(self.fcD(embed_xd))
        embed_xt = F.relu(self.fcT(xt))

        x = torch.cat([embed_xd, embed_xt], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        #outputs = []
        #outputs.append(x)
        return x

def train_CNNTansDTI2D(model, optimizer, device, train_loader):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    #features = features.to(device)

    for batch_idx, (fea, labels) in enumerate(train_loader):
#    for batch_idx, (fea, labels) in tqdm(train_loader):
        optimizer.zero_grad()
        fea = fea.to(device)
        labels = labels.to(device)
        outputs = model(fea)
        loss_train = 0
        loss_train = loss_train + criterion(outputs, torch.squeeze(labels))
        loss_train.backward()
        optimizer.step()

def test_CNNTansDTI2D(model, device, labels, test_loader):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        ii = 0
        for batch_idx, (fea, labels) in enumerate(test_loader):
            fea = fea.to(device)
            labels = labels.to(device)
            outputs = model(fea)

            if ii == 0:
                outputss = outputs
                labelss = labels
                ii = 1
            else:
                outputss = torch.cat((outputss, outputs), 0)
                labelss = torch.cat((labelss, labels), 0)
    print(outputss)
    print(labelss)
    outputs=[]
    outputs.append(outputss)
    test_probabilitys = []
    test_labels = []
    for ii in range(len(outputs)):
        output = outputs[ii]

        probability = nn.functional.softmax(output, dim=1)
        probability = probability.cpu()
        labelss = labelss.cpu()

        test_probabilitys.append(probability)
        test_labels.append(labelss)

    return test_probabilitys, test_labels


class RBMTansDTIES(nn.Module):
    def __init__(self,  in_t_feats):
        super(RBMTansDTIES, self).__init__()
        embedding_size = 768
        num_filters = 32
        dropout = 0.5
        k = 4
        n_layers = 1
        n_classes = 2
        n_hidden = 200
        in_d_feats = 768

        self.max_seq_len = 16
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.d_frames = self.max_seq_len
        self.d_dims = embedding_size

        self.dropout = nn.Dropout(p=dropout)

        self.layers1 = nn.ModuleList()
        self.layers1.append(nn.Linear(in_d_feats + n_hidden, n_hidden))
        for i in range(n_layers - 1):
            self.layers1.append(nn.Linear(n_hidden, n_hidden))
        self.layers1.append(nn.Linear(n_hidden, n_hidden))

        self.layers2 = nn.ModuleList()
        self.layers2.append(nn.Linear(in_t_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers2.append(nn.Linear(n_hidden, n_hidden))
        self.layers2.append(nn.Linear(n_hidden, n_hidden))

        #self.fc5 = nn.Linear(in_d_feats * self.max_seq_len + in_t_feats, n_hidden)
        self.fc5 = nn.Linear(n_hidden * k, n_hidden)
        self.fc6 = nn.Linear(n_hidden * 2, n_hidden)
        #self.fc7 = nn.Linear(n_hidden, n_hidden)
        #self.fc8 = nn.Linear(n_hidden, n_hidden)
        self.fc9 = nn.Linear(n_hidden, n_classes)
        self.k = k
        self.m = torch.nn.AdaptiveMaxPool2d((k, 768))

    def forward(self, fea):
        #print(fea.shape)

        drug_dim = self.d_frames * self.d_dims
        xd = fea[:, 0:drug_dim]
        xt = fea[:, drug_dim:fea.shape[1]]
        xd = xd.reshape(xd.shape[0], self.d_frames, self.d_dims)
        xd = self.m(xd)
        xd = xd.permute(1, 0, 2)
        # print(xd.shape)
        outputs = []
        bbb = 0
        ccc = []
        #print('xt')
       # print(xt.shape)

        for i, layer in enumerate(self.layers2):
            if i != 0:
               xt = self.dropout(xt)
            xt = F.relu(layer(xt))

        for ii in range(self.k):
            features = xd[ii, :, :]
            features = features.squeeze(0)
            # print(features.shape)
            #features = features.to(self.device)

            x = torch.cat((features, xt), 1)

            for i, layer in enumerate(self.layers1):
                if i != 0:
                    x = self.dropout(x)
                x = F.relu(layer(x))

            #xtg_ii = torch.cat((xt, x), 1)
            #xtg_ii = F.relu(self.fc6(xtg_ii))
            #xtg_ii = F.relu(self.fc7(xtg_ii))
            xtg_ii = self.fc9(x)
            outputs.append(xtg_ii)

            #bbb = bbb + x

            if ii == 0:
                bbb = x
            else:
                bbb = torch.cat((bbb, x), 1)


        #outputs = []
       # bbb = fea
        bbb = F.relu(self.fc5(bbb))
        #bbb = torch.cat((xt, bbb), 1)
        #bbb = F.relu(self.fc6(bbb))
        #bbb = F.relu(self.fc7(bbb))
        #bbb = F.relu(self.fc8(bbb))
        #bbb = F.relu(bbb/self.k)
        bbb = self.fc9(bbb)
        outputs.append(bbb)

        return outputs

def train_RBMTansDTIES(model, optimizer, device, train_loader):
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    loss_train = 0
    for batch_idx, (feas, labels) in enumerate(train_loader):
        #print(target_feas)
        optimizer.zero_grad()
        feas = feas.to(device)
        labels = labels.to(device)
        outputs = model(feas)
        '''
        loss_train = 10000000000000
        for ii in range(len(outputs) - 1):
            output = outputs[ii]
            this_loss = criterion(output, torch.squeeze(labels))
            if loss_train > this_loss:
                loss_train = this_loss
        '''
        #loss_train = loss_train + criterion(outputs[len(outputs) - 1], torch.squeeze(labels))
        for ii in range(len(outputs) ):
            output = outputs[ii]
            loss_train = loss_train + criterion(output, torch.squeeze(labels))

        loss_train.backward()
        optimizer.step()

def test_RBMTansDTIES(model, device, labels, test_loader):
    model.eval()
    # features = features.to(device)
    with torch.no_grad():
        ii = 0
        for batch_idx, (feas, labels) in enumerate(test_loader):
            feas = feas.to(device)
            labels = labels.to(device)
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
