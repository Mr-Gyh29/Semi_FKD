
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
import numpy as np

class Server(object):
    def __init__(self,args, global_model,Loaders_train, Loaders_local_test, Loader_global_test, logger, device):
        self.global_model = global_model
        self.args = args
        self.Loaders_train = Loaders_train
        self.Loaders_local_test = Loaders_local_test
        self.global_testloader = Loader_global_test
        self.logger = logger
        self.device = device
        self.LocalModels = []
        self.h = self.generate_channel_gain(args.num_clients, args.variance)
        
    def global_test_accuracy(self):
        self.global_model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.global_testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            _,p = self.global_model(X)
            y_pred = p.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt
    
    
    def Save_CheckPoint(self, save_path):
        torch.save(self.global_model.state_dict(), save_path)
    
    def Allocation_resource(self, idle_clients):
        bandwidth_per_client = [self.args.total_bandwidth / len(idle_clients) for _ in range(len(idle_clients))]
        frequency_per_client = [self.args.clock_frequency for _ in range(len(idle_clients))]

        return bandwidth_per_client, frequency_per_client
    
    
    def generate_channel_gain(self,num_users, variances):
        """
        生成多个用户与基站之间的信道增益 h_k。
        
        参数:
        num_users: 用户数量
        variances: 信道增益方差列表，长度应与 num_users 相同
        
        返回:
        h: 信道增益数组，形状为 (num_users,)
        """
        h = np.zeros(num_users, dtype=complex)
        for k in range(num_users):
            real_part = np.random.normal(0, np.sqrt(variances[k]))
            imag_part = np.random.normal(0, np.sqrt(variances[k]))
            h[k] = real_part + 1j * imag_part
        return h