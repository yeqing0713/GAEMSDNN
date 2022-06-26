from  sklearn.model_selection import KFold
import torch
from random import shuffle
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, ClusterCentroids, NearMiss

class CCrossExpController:
    def __init__(self, crosstype, drugFeatureVectors, targetFeatureVectors, Y, random_state, k_fold, ratio, is_SMOTE = False):
        train_indexs = []
        test_indexs = []
        if crosstype == 'CVD':
            self.drugFeatureVectors = drugFeatureVectors
            self.targetFeatureVectors = targetFeatureVectors
            self.Y = Y
            train_indexs, test_indexs = self.get_train_test_data_cross(self.drugFeatureVectors, k_fold, random_state)
        if crosstype == 'CVT':
            self.drugFeatureVectors = targetFeatureVectors
            self.targetFeatureVectors = drugFeatureVectors
            self.Y = Y.t()
            train_indexs, test_indexs = self.get_train_test_data_cross(self.drugFeatureVectors, k_fold, random_state)
        if crosstype == 'CVP':
            fea, labels = self.generate_data_by_drug_target(drugFeatureVectors, targetFeatureVectors, Y, ratio)
            train_indexs, test_indexs = self.get_train_test_data_cross(fea, k_fold, random_state)
            labels = torch.as_tensor(labels, dtype=torch.long)
            self.fea = fea
            self.labels = labels

        self.train_indexs = train_indexs
        self.test_indexs = test_indexs
        self.ratio = ratio
        self.k_fold = k_fold
        self.crosstype = crosstype
        self.is_SMOTE = is_SMOTE

    def get_train_test_cross_k(self, k):
        if self.crosstype == 'CVP':
            return self.get_train_test_cross_CVP(k)
        if self.crosstype == 'CVD':
            return self.get_train_test_cross_CVD(k)
        if self.crosstype == 'CVT':
            return self.get_train_test_cross_CVD(k)

    def get_train_test_cross_CVP(self, k):
        train_index = self.train_indexs[k]
        test_index = self.test_indexs[k]
        train_features = self.fea[train_index]
        train_labels = self.labels[train_index]
        test_features = self.fea[test_index]
        test_labels = self.labels[test_index]

        if self.is_SMOTE == True:
            smo = ADASYN(random_state=42)
            train_features, train_labels = smo.fit_sample(train_features, train_labels)
            train_features = torch.from_numpy(train_features)
            train_labels = torch.from_numpy(train_labels)

        train_index = np.arange(0, train_features.shape[0], 1)
        test_index = np.arange(train_features.shape[0], train_features.shape[0] + test_features.shape[0])
        features = torch.cat((train_features, test_features), 0)
        labels = torch.cat((train_labels, test_labels), 0)
        labels = torch.squeeze(labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        train_labels = torch.squeeze(train_labels)
        train_labels = torch.as_tensor(train_labels, dtype=torch.long)

        return features, labels, train_features, train_labels, test_features, test_labels, train_index, test_index

    def get_train_test_cross_CVD(self, k):
        train_index = self.train_indexs[k]
        test_index = self.test_indexs[k]

        train_drug_fea = self.drugFeatureVectors[train_index]
        train_drug_Y = self.Y[train_index]

        test_drug_fea = self.drugFeatureVectors[test_index]
        test_drug_Y = self.Y[test_index]
        train_features, train_labels = self.generate_data_by_drug_target(train_drug_fea, self.targetFeatureVectors, train_drug_Y, self.ratio)

        if self.is_SMOTE == True:
            smo = ADASYN(random_state=42)
            train_features, train_labels = smo.fit_sample(train_features, train_labels)
            train_features = torch.from_numpy(train_features)
            train_labels = torch.from_numpy(train_labels)

        test_features, test_labels = self.generate_data_by_drug_target(test_drug_fea, self.targetFeatureVectors, test_drug_Y, self.ratio)
        train_index = np.arange(0, train_features.shape[0], 1)
        test_index = np.arange(train_features.shape[0], train_features.shape[0] + test_features.shape[0])

        features = torch.cat((train_features, test_features), 0)
        labels = torch.cat((train_labels, test_labels), 0)
        labels = torch.squeeze(labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        train_labels = torch.squeeze(train_labels)
        train_labels = torch.as_tensor(train_labels, dtype=torch.long)
        return features, labels, train_features, train_labels, test_features, test_labels, train_index, test_index

    def get_train_test_data_cross(self, fea, k_fold, random_state):
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        train_indexs = []
        test_indexs = []
        for train_index, test_index in kf.split(fea):
            #print(train_index)
            train_indexs.append(train_index)
            test_indexs.append(test_index)
        return train_indexs, test_indexs

    def generate_data_by_drug_target(self,drugFeatureVectors, targetFeatureVectors, Y, ratio):
        a = Y.shape
        num_total = a[1] * a[0]
        num_DTI = len(Y[Y == 1])
        num_unDTI = num_total - num_DTI
        inx_DTI = np.where(Y == 1)
        inx_unDTI = np.where(Y == 0)

        drug_feas = drugFeatureVectors[inx_DTI[0], :]
        target_fea = targetFeatureVectors[inx_DTI[1], :]
        pos_feas = torch.cat((drug_feas, target_fea), 1)
        pos_Y = torch.zeros((num_DTI, 1)) + 1

        inx1 = [i for i in range(num_unDTI)]
        shuffle(inx1)
        if ratio != 0:
            drug_feas = drugFeatureVectors[inx_unDTI[0][inx1[0:num_DTI * ratio]], :]
            target_fea = targetFeatureVectors[inx_unDTI[1][inx1[0:num_DTI * ratio]], :]
        else:
            drug_feas = drugFeatureVectors[inx_unDTI[0], :]
            target_fea = targetFeatureVectors[inx_unDTI[1], :]

        neg_feas = torch.cat((drug_feas, target_fea), 1)
        neg_Y = torch.zeros((neg_feas.shape[0], 1))

        data = torch.cat((pos_feas, neg_feas), 0)
        labels = torch.cat((pos_Y, neg_Y), 0)
        labels = labels.squeeze(1)
        return data, labels

    def generate_data_by_drug_target_with_inxs(self,drugFeatureVectors, targetFeatureVectors, Y, ratio):
        a = Y.shape
        num_total = a[1] * a[0]
        num_DTI = len(Y[Y == 1])
        num_unDTI = num_total - num_DTI
        inx_DTI = np.where(Y == 1)
        inx_unDTI = np.where(Y == 0)

        drug_feas = drugFeatureVectors[inx_DTI[0], :]
        target_fea = targetFeatureVectors[inx_DTI[1], :]
        pos_feas = torch.cat((drug_feas, target_fea), 1)
        pos_Y = torch.zeros((num_DTI, 1)) + 1

        inx1 = [i for i in range(num_unDTI)]
        if ratio != 0:
            drug_feas = drugFeatureVectors[inx_unDTI[0][inx1[0:num_DTI * ratio]], :]
            target_fea = targetFeatureVectors[inx_unDTI[1][inx1[0:num_DTI * ratio]], :]
        else:
            drug_feas = drugFeatureVectors[inx_unDTI[0], :]
            target_fea = targetFeatureVectors[inx_unDTI[1], :]

        neg_feas = torch.cat((drug_feas, target_fea), 1)
        neg_Y = torch.zeros((neg_feas.shape[0], 1))

        data = torch.cat((pos_feas, neg_feas), 0)
        labels = torch.cat((pos_Y, neg_Y), 0)
        labels = labels.squeeze(1)
        return data, labels, inx_DTI, inx_unDTI