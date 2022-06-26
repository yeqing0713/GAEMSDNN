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
from sklearn.feature_selection import VarianceThreshold
import SurDimReductionController

def DTI_one_random(df_path, tf_path, Y_path, random_state, exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, graph_type, neg_ratio, k, sub_ratio, is_SMOTE = False, is_gcn = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 30000
    k_fold = 5
    drugFeatureVectors, targetFeatureVectors, Y = LoadData.load_data(df_path, tf_path, Y_path)
    #sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    #drugFeatureVectors = sel.fit_transform(drugFeatureVectors)
    #targetFeatureVectors = sel.fit_transform(targetFeatureVectors)
    #print(drugFeatureVectors.shape)
    #print(targetFeatureVectors.shape)
    #drugFeatureVectors = torch.from_numpy(drugFeatureVectors)
    #targetFeatureVectors = torch.from_numpy(targetFeatureVectors)

    drugFeatureVectors, targetFeatureVectors = DimReductionController.DR_preprocess_main(DR_type, drugFeatureVectors, targetFeatureVectors, Y, device, batch_size)

    cec = CrossExpController.CCrossExpController(exp_type, drugFeatureVectors, targetFeatureVectors, Y, random_state, k_fold, neg_ratio, is_SMOTE)

    totalss = 0
    for kk in range(k_fold):
        features, labels, train_features, train_labels, test_features, test_labels, train_index, test_index = cec.get_train_test_cross_k(kk)
        features, train_features, test_features = SurDimReductionController.DR_preprocess_main(features, labels, train_features, train_labels, test_features, test_labels, model_type)
        g = []
        if is_gcn == True:
            g = GCNGraphcreate.create_graph(graph_type, train_features, train_labels, features, labels).to(device)

        in_feats = features.shape[1]
        n_class = 2
        model = ModelController.create_model('RBMNet_3_d', in_feats, hider_dim, n_class, layer_UU, F.relu, drop_ratio, device, g).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        train_dataset = DNNDaset.CDNNDataset(train_features, train_labels)
        test_dataset = DNNDaset.CDNNDataset(test_features, test_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        matrics = []
        features = features.to(device)
        labels = labels.to(device)
        for epoch in range(400):
            ModelController.train_model('RBMNet_3_d', model, optimizer, features, device, labels, train_index, train_loader)
            if epoch % 20 == 0 and epoch >= 20:
                test_probabilitys, test_labels = ModelController.test_model('RBMNet_3_d', model, features, device, labels, test_index, test_loader)
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
