import torch
import LoadData
import DimReductionController
import CrossExpController
import GCNGraphcreate
import ModelController
import torch.nn.functional as F
import torch.optim as optim
import DNNDaset
import utils
import numpy
import SurDimReductionController
import RandomSubspace
from sklearn.feature_selection import VarianceThreshold

def DTI_one_random(df_path, tf_path, Y_path, random_state, exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, graph_type, v, k, sub_ratio, is_SMOTE = False, is_gcn = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4096
    k_fold = 5
    neg_ratio = 5
    drugFeatureVectors, targetFeatureVectors, Y = LoadData.load_data(df_path, tf_path, Y_path)
    #sel = VarianceThreshold(threshold=(v * (1 - v)))
    #drugFeatureVectors = sel.fit_transform(drugFeatureVectors)
    #targetFeatureVectors = sel.fit_transform(targetFeatureVectors)
    #drugFeatureVectors = torch.from_numpy(drugFeatureVectors)
    #targetFeatureVectors = torch.from_numpy(targetFeatureVectors)
    drugFeatureVectors, targetFeatureVectors = DimReductionController.DR_preprocess_main(DR_type, drugFeatureVectors, targetFeatureVectors, Y, device, batch_size)
    drugFeatureVectors.to("cpu")
    targetFeatureVectors.to("cpu")
    cec = CrossExpController.CCrossExpController(exp_type, drugFeatureVectors, targetFeatureVectors, Y, random_state, k_fold, neg_ratio, is_SMOTE)

    dim = drugFeatureVectors.shape[1] + targetFeatureVectors.shape[1]
    in_feats = int(dim * sub_ratio)
    rss = RandomSubspace.get_k_random(dim, k, in_feats)
    totalss = 0
    for kk in range(k_fold):
        features, labels, train_features, train_labels, test_features, test_labels, train_index, test_index = cec.get_train_test_cross_k(kk)

        #g = []
        #if is_gcn == True:
        #    g = GCNGraphcreate.create_graph(graph_type, train_features, train_labels, features, labels).to(device)

        #in_feats = features.shape[1]
        n_class = 2
        #features = features.to(device)
        fea_subspaces = RandomSubspace.get_k_random_subspaces(features, rss)

        #for ii in range(len(fea_subspaces)):
        #    print(fea_subspaces[ii].shape)

        gs = []
        if is_gcn == True:
            for ii in range(len(fea_subspaces)):
                train_features_s = fea_subspaces[ii]
                train_features_s = train_features_s[train_index]
                features_s = fea_subspaces[ii]
                #g = GCNGraphcreate.create_graph(graph_type, train_features_s, train_labels, features_s, labels).to(device)
                g = []
                gs.append(g)

        model = ModelController.create_model(model_type, in_feats, hider_dim, n_class, layer_UU, F.relu, drop_ratio, device, gs, k).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        train_dataset = DNNDaset.CDNNDatasetMS(train_features, train_labels)
        test_dataset = DNNDaset.CDNNDatasetMS(test_features, test_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        matrics = []
        #
        labels = labels.to(device)
        for epoch in range(600):
            ModelController.train_model(model_type, model, optimizer, fea_subspaces, device, labels, train_index, train_loader)
            if epoch % 20 == 0 and epoch >= 20:
                test_probabilitys, test_labels = ModelController.test_model(model_type, model, fea_subspaces, device, labels, test_index, test_loader)
                matric = utils.probability2matrics(test_probabilitys, test_labels)
                matrics.append(matric)

        [aa, bb] = matrics[0].shape
        matrics1 = numpy.zeros((len(matrics), aa, bb))
        for iii in range(len(matrics)):
            matrics1[iii] = matrics[iii]

        totals = matrics1
        totalss = totalss + totals

    myarray = numpy.array(totalss)
    myarray = myarray/k_fold
    return myarray
