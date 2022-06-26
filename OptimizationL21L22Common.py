import numpy as np
import torch
import torch.functional

def OptimizationL21L22Common(X, Y, device):
    X = torch.as_tensor(X, dtype=torch.float).to(device)
    #Y = torch.as_tensor(Y, dtype=torch.float).to(device)
    print(Y.shape)
    Y = torch.nn.functional.one_hot(Y, num_classes=2)
    print(Y.shape)
    Y = torch.as_tensor(Y, dtype=torch.float).to(device)
    lamb = torch.as_tensor(0.1, dtype=torch.float).to(device)
    size = X.shape
    n = size[0]
    m = size[1]
    I1 = np.eye(m)
    print(I1)
    I1 = torch.from_numpy(I1)
    I1 = torch.as_tensor(I1, dtype=torch.float).to(device)
    Ut = np.identity(m)
    Ut = torch.from_numpy(Ut)
    Ut = torch.as_tensor(Ut, dtype=torch.float).to(device)


    Xt =  X.t()
    XtY = Xt.mm(Y)
    XtX = Xt.mm(X)
    for t in range(1, 15):
        aa = Ut.mm(XtX) + lamb * I1
        print(aa)

        bb = torch.inverse(aa)
        bb = bb.mm(Ut)
        W = bb.mm(XtY)

        W_2 = W.mul(W)
        Ut = torch.diag(torch.sqrt(torch.sum(W_2, 1)) * 2)

    return W