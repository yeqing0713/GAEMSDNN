from sklearn.model_selection import KFold
import torch
from random import shuffle
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, ClusterCentroids, NearMiss

class CCrossExpController:
    def __init__(self, crosstype, drugFeatureVectors, targetFeatureVectors, Y, random_state, k_fold, ratio, generator_num,is_SMOTE = False):
        self.generator_num = generator_num
        train_indexs = []
        test_indexs = []
        [num, dim] = drugFeatureVectors.shape
        drug_num = int(num/generator_num)
        print(drug_num)
        drug_feaV_one = drugFeatureVectors[0:drug_num, :]
        drug_Y_one = Y[0:drug_num, :]

        if crosstype == 'CVD':
            self.drugFeatureVectors = drugFeatureVectors
            self.targetFeatureVectors = targetFeatureVectors
            self.Y = Y
            train_indexs, test_indexs = self.get_train_test_data_cross_CVD(drug_feaV_one, k_fold, random_state, generator_num, drug_num)
        if crosstype == 'CVT':
            self.drugFeatureVectors = targetFeatureVectors
            self.targetFeatureVectors = drugFeatureVectors
            self.Y = Y.t()
            train_indexs, test_indexs = self.get_train_test_data_cross_CVT(self.drugFeatureVectors, k_fold, random_state)
        if crosstype == 'CVP':
            [inx_DTI, inx_unDTI,  inx1] = self.generate_inx_CVP(drug_Y_one)
            feas = []
            labels = []
            for ii in range(generator_num):
                start_inx = ii * drug_num
                end_inx = (ii + 1) * drug_num
                drug_feaV_one = drugFeatureVectors[start_inx:end_inx, :]
                drug_Y_one = Y[start_inx:end_inx, :]
                [fea, label] = self.generate_data_by_drug_target_CVP(drug_feaV_one, targetFeatureVectors, drug_Y_one, inx_DTI, inx_unDTI, inx1, ratio)
                label = torch.as_tensor(label, dtype=torch.long)
                feas.append(fea)
                labels.append(label)

            train_indexs, test_indexs = self.get_train_test_data_cross_CVP(feas[0], k_fold, random_state)
            self.feas = feas
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
        flag = 0
        train_feas = []
        train_labls = []
        test_feas = []
        test_labels = []
        for ii in range(self.generator_num):
            fea_ii = self.feas[ii]
            label_ii = self.labels[ii]
            train_fea_ii = fea_ii[train_index]
            test_fea_ii = fea_ii[test_index]
            train_label_ii = label_ii[train_index]
            test_label_ii = label_ii[test_index]
            if flag == 0:
                flag = 1
                train_feas = train_fea_ii
                train_labls = train_label_ii
                test_feas = test_fea_ii
                test_labels = test_label_ii
            else:
                train_feas = torch.cat((train_feas, train_fea_ii), 0)
                train_labls = torch.cat((train_labls, train_label_ii), 0)
                test_feas = torch.cat((test_feas, test_fea_ii), 0)
                test_labels = torch.cat((test_labels, test_label_ii), 0)

        train_features = train_feas
        train_labels = train_labls
        test_features = test_feas
        test_labels = test_labels

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

    def get_train_test_data_cross_CVP(self, fea, k_fold, random_state):
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        train_indexs = []
        test_indexs = []
        for train_index, test_index in kf.split(fea):
            train_indexs.append(train_index)
            test_indexs.append(test_index)
        return train_indexs, test_indexs

    def get_train_test_data_cross_CVT(self, fea, k_fold, random_state):
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        train_indexs = []
        test_indexs = []
        for train_index, test_index in kf.split(fea):
            train_indexs.append(train_index)
            test_indexs.append(test_index)
        return train_indexs, test_indexs

    def get_train_test_data_cross_CVD(self, fea, k_fold, random_state, generator_num, drug_num):
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        train_indexs = []
        test_indexs = []
        for train_index, test_index in kf.split(fea):
            train_gen_index = train_index
            train_gen_index = torch.from_numpy(train_gen_index)
            t_train_index = train_index
            t_train_index = torch.from_numpy(t_train_index)
            test_gen_inx = test_index
            t_test_inx = test_index
            test_gen_inx = torch.from_numpy(test_gen_inx)
            t_test_inx = torch.from_numpy(t_test_inx)
            for jj in range(generator_num - 1):
                t_train_index = t_train_index + drug_num
                torch.cat((train_gen_index, t_train_index), 0)
                t_test_inx = t_test_inx + drug_num
                torch.cat((test_gen_inx, t_test_inx), 0)

            train_gen_index = torch.as_tensor(train_gen_index, dtype=torch.long)
            test_gen_inx = torch.as_tensor(test_gen_inx, dtype=torch.long)
            train_indexs.append(train_gen_index)
            test_indexs.append(test_gen_inx)
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

    def generate_inx_CVP(self, Y):
        a = Y.shape
        num_total = a[1] * a[0]
        num_DTI = len(Y[Y == 1])
        num_unDTI = num_total - num_DTI
        inx_DTI = np.where(Y == 1)
        inx_unDTI = np.where(Y == 0)

        inx1 = [i for i in range(num_unDTI)]
        shuffle(inx1)
        return inx_DTI, inx_unDTI, inx1

    def generate_data_by_drug_target_CVP(self,drugFeatureVectors, targetFeatureVectors, Y, inx_DTI, inx_unDTI, inx1, ratio):

        num_DTI = len(inx_DTI[0])
        drug_feas = drugFeatureVectors[inx_DTI[0], :]
        target_fea = targetFeatureVectors[inx_DTI[1], :]
        pos_feas = torch.cat((drug_feas, target_fea), 1)
        pos_Y = torch.zeros((num_DTI, 1)) + 1

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