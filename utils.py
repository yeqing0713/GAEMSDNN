
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import numpy
import torch

def probability2acc(probabilitys, labels):
    accs = [0.0] * len(probabilitys)
    for ii in range(len(probabilitys)):
        probability = probabilitys[ii]
        label = labels[ii]
        pre = probability[:, 1]
        pred = probability.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accs[ii] = accuracy_score(label.cpu(), pred.cpu()) * 100
    return accs

def probability2matrics(probabilitys, labels):
    aucs = [0.0] * len(probabilitys)
    AUPRs = [0.0] * len(probabilitys)
    accs = [0.0] * len(probabilitys)
    f1ss = [0.0] * len(probabilitys)
    pcis = [0.0] * len(probabilitys)
    rcss = [0.0] * len(probabilitys)
    MCCs = [0.0] * len(probabilitys)
    matrics = numpy.zeros((7, len(probabilitys)))

    for ii in range(len(probabilitys)):
        probability = probabilitys[ii]
        label = labels[ii]
        pre = probability[:, 1]
        pred = probability.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        aucs[ii] = roc_auc_score(label.cpu(), pre.cpu()) * 100
        precision, recall, _thresholds = precision_recall_curve(label.cpu(), pre.cpu())
        AUPRs[ii] = auc(recall, precision) * 100
        accs[ii] = accuracy_score(label.cpu(), pred.cpu()) * 100
        f1ss[ii] = f1_score(label.cpu(), pred.cpu()) * 100
        pcis[ii] = precision_score(label.cpu(), pred.cpu()) * 100
        rcss[ii] = recall_score(label.cpu(), pred.cpu()) * 100
        MCCs[ii] = matthews_corrcoef(label.cpu(), pred.cpu()) * 100

    matrics[1] = aucs
    matrics[0] = AUPRs
    matrics[2] = pcis
    matrics[3] = accs
    matrics[4] = f1ss
    matrics[5] = rcss
    matrics[6] = MCCs

    return matrics

def probability2drugtargetinx(probabilitys, test_drug_target_ids ,labels, Y):
    drug_idxs = [0] * 10
    target_idxs = [0] * 10
    for ii in range(len(probabilitys)):
        probability = probabilitys[ii]
        label = labels[ii]
        pre = probability[:, 1]
        print(sklearn.metrics.roc_auc_score(label.cpu(), pre.cpu()) * 100)
        sorted, indices = torch.sort(pre, descending=True)
        k = 0
        for jj in range(label.shape[0]):
            if label[indices[jj]] == 0:
                drug_inx = int(int(test_drug_target_ids[indices[jj]])/(Y.shape[1]))
                target_inx = int(test_drug_target_ids[indices[jj]])%(Y.shape[1])
                drug_idxs[k] = drug_inx + 1
                target_idxs[k] = target_inx + 1
                k = k + 1
                if k == 10:
                    break

    return drug_idxs, target_idxs

def probability2drugtargetname(probabilitys, inx_DTI, inx_unDTI ,labels):
    drug_idxs = [0] * 100
    target_idxs = [0] * 100
    labelx = torch.zeros([100, 1])
    prex = torch.zeros([100, 1])
    for ii in range(len(probabilitys)):
        probability = probabilitys[ii]
        label = labels[ii]
        pre = probability[:, 1]
        #print(sklearn.metrics.roc_auc_score(label.cpu(), pre.cpu()) * 100)
        sorted, indices = torch.sort(pre, descending=True)
        k = 0
        for jj in range(label.shape[0]):
            if indices[jj] >= len(inx_DTI[0]):
                inx = indices[jj] - len(inx_DTI[0])
                drug_inx = inx_unDTI[0][inx]
                target_inx = inx_unDTI[1][inx]
            else:
                inx = indices[jj]
                drug_inx = inx_DTI[0][inx]
                target_inx = inx_DTI[1][inx]

            labelx[k] = label[indices[jj]]
            prex[k] = pre[indices[jj]]
            drug_idxs[k] = drug_inx
            target_idxs[k] = target_inx

            k = k+1
            if k == 100:
                break

    labelx = labelx.numpy()
    prex = prex.numpy()
    return drug_idxs, target_idxs, labelx, prex




