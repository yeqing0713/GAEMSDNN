from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression , mutual_info_regression, f_classif, mutual_info_classif
import OptimizationL21L22Common

def L21_FS(features, labels, train_features, train_labels, test_features, test_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = train_features.shape[1]
    re_dim = int(dim * 0.5)
    print(train_features.shape)
    W = OptimizationL21L22Common.OptimizationL21L22Common(train_features, train_labels, device)
    W = torch.mean(torch.abs(W), 1)
    sorted, indices = torch.sort(W, descending=True)
    features = features[:, indices[0:re_dim]]
    train_features = train_features[:, indices[0:re_dim]]
    test_features = test_features[:, indices[0:re_dim]]
    return features, train_features, test_features

def SKlearn_L1_FS(features, labels, train_features, train_labels, test_features, test_labels):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_features, train_labels)
    model = SelectFromModel(lsvc, prefit=True)
    features = model.transform(features)
    train_features = model.transform(train_features)
    test_features = model.transform(test_features)
    features = torch.from_numpy(features)
    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    return features, train_features, test_features

def SKlearn_KBESt_FS(features, labels, train_features, train_labels, test_features, test_labels):
    dim = train_features.shape[1]
    re_dim = int(dim * 0.1)
    selector = SelectKBest(mutual_info_classif, k=re_dim)
    selector.fit(features, labels)
    print(features.shape)
    features = selector.transform(features)
    print(features.shape)
    train_features = selector.transform(train_features)
    test_features = selector.transform(test_features)
    features = torch.from_numpy(features)
    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    return features, train_features, test_features

def SKlearn_KBESt_FS1(features, labels, train_features, train_labels, test_features, test_labels):
    dim = train_features.shape[1]
    re_dim = int(dim * 0.1)
    selector = SelectKBest(chi2, k=re_dim)
    selector.fit(features, labels)
    print(features.shape)
    features = selector.transform(features)
    print(features.shape)
    train_features = selector.transform(train_features)
    test_features = selector.transform(test_features)
    features = torch.from_numpy(features)
    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    return features, train_features, test_features

def SKlearn_KBESt_FS3(features, labels, train_features, train_labels, test_features, test_labels):
    dim = train_features.shape[1]
    re_dim = int(dim * 0.1)
    selector = SelectKBest(mutual_info_classif, k=re_dim)
    selector.fit(train_features, train_labels)
    print(features.shape)
    features = selector.transform(features)
    print(features.shape)
    train_features = selector.transform(train_features)
    test_features = selector.transform(test_features)
    features = torch.from_numpy(features)
    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    return features, train_features, test_features

def SKlearn_KBESt_FS4(features, labels, train_features, train_labels, test_features, test_labels):
    dim = train_features.shape[1]
    re_dim = int(dim * 0.5)
    selector = SelectKBest(chi2, k=re_dim)
    selector.fit(train_features, train_labels)
    print(features.shape)
    features = selector.transform(features)
    print(features.shape)
    train_features = selector.transform(train_features)
    test_features = selector.transform(test_features)
    features = torch.from_numpy(features)
    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    return features, train_features, test_features

def DR_preprocess_main(features, labels, train_features, train_labels, test_features, test_labels, sur_dim_type):
    if sur_dim_type == 'L21_FS':
        return L21_FS(features, labels, train_features, train_labels, test_features, test_labels)
    elif sur_dim_type == 'SKlearn_L1_FS':
        return SKlearn_L1_FS(features, labels, train_features, train_labels, test_features, test_labels)
    elif sur_dim_type == 'SKlearn_KBESt_FS':
        return SKlearn_KBESt_FS(features, labels, train_features, train_labels, test_features, test_labels)
    elif sur_dim_type == 'SKlearn_KBESt_FS1':
        return SKlearn_KBESt_FS1(features, labels, train_features, train_labels, test_features, test_labels)
    elif sur_dim_type == 'SKlearn_KBESt_FS3':
        return SKlearn_KBESt_FS3(features, labels, train_features, train_labels, test_features, test_labels)
    elif sur_dim_type == 'SKlearn_KBESt_FS4':
        return SKlearn_KBESt_FS4(features, labels, train_features, train_labels, test_features, test_labels)
