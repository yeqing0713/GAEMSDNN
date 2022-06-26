import DNNModels
import GCNModels
import DNNModelsMS

DNNModels_Types = ['AutoEncoder', 'RBMGoogleNet', 'RBMGoogleNetRatio', 'RBMNetRatio', 'RBMNet_2000', 'RBMNet_3', 'RBMNet_3_DM', 'RBMNet_3_MS']
GCNModels_Types = ['GCN_AutoEncoder', 'GCNNout', 'GCN', 'GCN_DNN', 'RES_GCN']
model_types = ['GCNTAGConvNout', 'GCNSAGEConvNout', 'GCNGINConvNout',  'GCNSGConvNout', 'GCNChebConvNout', 'GCNDotGatConvNout']
def create_model(model_type, in_feats, n_hidden = 'none', n_classes = 'none', n_layers = 'none', activation = 'none', dropout = 'none', device = 'none', g = 'none', k = 'none'):
    model = []
    if model_type == 'AutoEncoder':
        model = DNNModels.AutoEncoder(in_feats, n_hidden)
    elif model_type == 'RBMGoogleNet_MS':
        model = DNNModelsMS.RBMGoogleNet_MS(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'GCN_AutoEncoder':
        model = GCNModels.GCN_AutoEncoder(in_feats, n_hidden, activation, device, g)
    elif model_type == 'VGAEModel':
        model = GCNModels.VGAEModel(in_feats, n_hidden, activation, device, g)
    elif model_type == 'GCN_3_MS':
        model = DNNModelsMS.GCN_3_MS(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device, g)
    elif model_type == 'RBM_n_MS':
        model = DNNModelsMS.RBM_n_MS(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'RBM_n_MS_DL':
        model = DNNModelsMS.RBM_n_MS(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'RBM_n_MS_DL1':
        model = DNNModelsMS.RBM_n_MS(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'RBM_n_MS_D':
        model = DNNModelsMS.RBM_n_MS_D(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'RBM_n_MS_D2':
        model = DNNModelsMS.RBM_n_MS_D2(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'RBM_n_MSU':
        model = DNNModelsMS.RBM_n_MSU(in_feats, n_hidden, n_classes, n_layers, k, activation, dropout, device)
    elif model_type == 'RBMTansDTI':
        model = DNNModels.RBMTansDTI(in_feats)
    elif model_type == 'CNNTansDTI2D':
        model = DNNModels.CNNTansDTI2D(in_feats)
    elif model_type == 'RBMTansDTIES':
        model = DNNModels.RBMTansDTIES(in_feats)
    return model

def train_model(model_type, model, optimizer = 'none', features = 'none', device = 'none', labels = 'none', train_index = 'none', train_loader = 'none', matric = 'none', epoch = 'none'):
    if model_type == 'AutoEncoder':
        DNNModels.train_AutoEncoder(model, optimizer, train_loader, device)
    elif model_type == 'RBMGoogleNet_MS':
        DNNModelsMS.train_MS(model, optimizer, features, labels, train_index, device)
    elif model_type == 'GCN_AutoEncoder':
        GCNModels.train_GCN_AutoEncoder(model, optimizer, features, device)
    elif model_type == 'VGAEModel':
        GCNModels.train_VGAEModel(model, optimizer, features, device)
    elif model_type == 'GCN_3_MS':
        DNNModelsMS.train_MS1(model, optimizer, features, labels, train_index, device)
    elif model_type == 'RBM_n_MS':
        DNNModelsMS.train_MS1(model, optimizer, features, labels, train_index, device)
    elif model_type == 'RBM_n_MS_DL':
        DNNModelsMS.train_MS1_DL(model, optimizer, features, train_loader, device)
    elif model_type == 'RBM_n_MS_DL1':
        DNNModelsMS.train_MS1_DL(model, optimizer, features, train_loader, device)
    elif model_type == 'RBM_n_MS_D':
        DNNModelsMS.train_MS2(model, optimizer, features, labels, train_index, device)
    elif model_type == 'RBM_n_MS_D2':
        DNNModelsMS.train_MS3(model, optimizer, features, labels, train_index, device)
    elif model_type == 'RBM_n_MSU':
        DNNModelsMS.train_MS1_U(model, optimizer, features, labels, train_index, device)
    elif model_type == 'CNNTansDTI2D':
        DNNModels.train_CNNTansDTI2D(model, optimizer, device, train_loader)
    elif model_type == 'RBMTansDTI':
        DNNModels.train_RBMTansDTI(model, optimizer, device, train_loader)
    elif model_type == 'RBMTansDTIES':
        DNNModels.train_RBMTansDTIES(model, optimizer, device, train_loader)

def test_model(model_type, model, features, device = 'none', labels = 'none', idx_test = 'none', test_loader = 'none'):
    if model_type == 'AutoEncoder':
        return DNNModels.test_AutoEncoder(model, features, device)
    elif model_type == 'RBMGoogleNet_MS':
        return DNNModelsMS.test_MS(model, features, labels, idx_test)
    elif model_type == 'GCN_AutoEncoder':
        return GCNModels.test_GCN_AutoEncoder(model, features, device)
    elif model_type == 'VGAEModel':
        return GCNModels.test_VGAEModel(model, features, device)
    elif model_type == 'GCN_3_MS':
        return DNNModelsMS.test_MS1(model, features, labels, idx_test)
    elif model_type == 'RBM_n_MS':
        return DNNModelsMS.test_MS1(model, features, labels, idx_test)
    elif model_type == 'RBM_n_MS_DL':
        return DNNModelsMS.test_MS1_DL(model, features, labels, idx_test, device)
    elif model_type == 'RBM_n_MS_DL1':
        return DNNModelsMS.test_MS1_DL1(model, features, labels, device)
    elif model_type == 'RBM_n_MS_D':
        return DNNModelsMS.test_MS2(model, features, labels, idx_test)
    elif model_type == 'RBM_n_MS_D2':
        return DNNModelsMS.test_MS3(model, features, labels, idx_test)
    elif model_type == 'RBM_n_MSU':
        return DNNModelsMS.test_MS1_U(model, features, labels, idx_test)
    elif model_type == 'RBMTansDTI':
        return DNNModels.test_RBMTansDTI(model, device, labels, test_loader)
    elif model_type == 'RBMTansDTIES':
        return DNNModels.test_RBMTansDTIES(model, device, labels, test_loader)