import CNNModels


DNNModels_Types = ['AutoEncoder', 'RBMGoogleNet', 'RBMGoogleNetRatio', 'RBMNetRatio', 'RBMNet_2000', 'RBMNet_3', 'RBMNet_3_DM', 'RBMNet_3_MS']
GCNModels_Types = ['GCN_AutoEncoder', 'GCNNout', 'GCN', 'GCN_DNN', 'RES_GCN']
model_types = ['GCNTAGConvNout', 'GCNSAGEConvNout', 'GCNGINConvNout',  'GCNSGConvNout', 'GCNChebConvNout', 'GCNDotGatConvNout']
def create_model(model_type, in_d_feats = 'none',  in_t_feats = 'none',n_hidden = 'none', n_classes = 'none', n_layers = 'none', k = 'none', dropout = 'none', device = 'none'):
    model = []
    if model_type == 'DeepDTA':
        model = CNNModels.DeepDTA()
    elif model_type == 'DeepConvDTI':
        model = CNNModels.DeepConvDTI(in_d_feats)
    elif model_type == 'DeepDDTI':
        model = CNNModels.DeepDDTI(in_d_feats, in_t_feats)
    elif model_type == 'DeepTansDTI':
        model = CNNModels.DeepTansDTI(in_d_feats, in_t_feats)
    elif model_type == 'DeepTansDTIES':
        model = CNNModels.DeepTansDTIES(in_d_feats, in_t_feats, n_hidden, n_classes, n_layers, k, dropout, device)
    return model

def train_model(model_type, model, optimizer = 'none', device = 'none', train_loader = 'none', drug_feas = 'none', target_feas = 'none',  labels = 'none', train_index = 'none', ):
    if model_type == 'DeepDTA':
        CNNModels.train_DeepDTA(model, optimizer, device, train_loader)
    elif model_type == 'DeepConvDTI':
        CNNModels.train_DeepConvDTI(model, optimizer, device, train_loader)
    elif model_type == 'DeepDDTI':
        CNNModels.train_DeepDDTI(model, optimizer, device, train_loader)
    elif model_type == 'DeepTansDTI':
        CNNModels.train_DeepTansDTI(model, optimizer, device, train_loader)
        #CNNModels.train_DeepDDTI1(model, optimizer, drug_feas, target_feas, labels, device)
    elif model_type == 'DeepTansDTIES':
        CNNModels.train_MS1(model, optimizer, device, train_loader)


def test_model(model_type, model, drug_feas, target_feas, device = 'none', labels = 'none', idx_test = 'none', test_loader = 'none'):
    if model_type == 'DeepDTA':
        return CNNModels.test_DeepDTA(model, drug_feas, target_feas, labels, idx_test, device)
    elif model_type == 'DeepConvDTI':
        return CNNModels.test_DeepConvDTI(model, drug_feas, target_feas, labels, idx_test, device)
    elif model_type == 'DeepDDTI':
        return CNNModels.test_DeepDDTI(model, drug_feas, target_feas, labels, idx_test)
    elif model_type == 'DeepTansDTI':
        return CNNModels.test_DeepTansDTI(model, device, labels, test_loader)
    elif model_type == 'DeepTansDTIES':
        return CNNModels.test_MS1(model, device, labels, test_loader)
