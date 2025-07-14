import torch
import copy

# 计算准确率
def Accuracy(y,y_predict):
    leng = len(y)
    miss = 0
    for i in range(leng):
        if not y[i]==y_predict[i]:
            miss +=1
    return (leng-miss)/leng


def soft_predict(Z, temp):
    m, n = Z.shape
    Q = torch.zeros(m, n)
    Z_max = torch.max(Z / temp, dim=1, keepdim=True).values  # Stabilize by subtracting max
    Z_exp = torch.exp((Z / temp) - Z_max)
    Z_sum = torch.sum(Z_exp, dim=1, keepdim=True)
    Q = Z_exp / Z_sum
    return Q

def FedAvg(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
