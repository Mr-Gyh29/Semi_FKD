
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
        self.variance = [np.random.uniform(0, 1) for _ in range(args.num_clients)]
        self.h = self.generate_channel_gain(args.num_clients, self.variance)
        
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



        

    def Equal_B(self, idle_clients):
        # 计算每个用户的带宽和频率
        ratio = 1 / len(idle_clients)
        return ratio
    
    def compute_resource(self, ratio_B, ratio_F, idle_clients):
        # 计算每个用户的带宽和频率
        for client in range(self.args.num_clients):
            self.LocalModels[client].bandwidth = self.args.total_bandwidth * ratio_B[client]
            self.LocalModels[client].clock_frequency = self.args.clock_frequency * ratio_F[client]
        
    
    def generate_channel_gain(self,num_users, variances=1):
        """
        生成多个用户与基站之间的信道增益 h_k。
        
        参数:
        num_users: 用户数量
        variances: 信道增益方差列表，长度应与 num_users 相同
        
        返回:
        h: 信道增益数组，形状为 (num_users,)
        """
        print("variances: ", variances)
        h = np.zeros(num_users, dtype=complex)
        for k in range(num_users):
            real_part = np.random.normal(0, np.sqrt(variances[k]))
            imag_part = np.random.normal(0, np.sqrt(variances[k]))
            h[k] = real_part + 1j * imag_part
        print("h: ", np.abs(h)**2)
        return h
    
    def Tier_clients(self, tier_method, num_tiers):
        # 随机分层, 将所有用户随机分配到某一层，最后返回这个记录每层有哪些用户的字典,字典key表示第几层，value表示该层的用户列表
        if tier_method == 'Random':
            # 规定每层有多少用户
            clients_per_tier = np.random.multinomial(self.args.num_clients, [1/num_tiers]*num_tiers)

            assert sum(clients_per_tier) == self.args.num_clients, "每层用户数之和必须等于总用户数"

            clients = list(range(self.args.num_clients))
            np.random.shuffle(clients)

            tiered_clients = {}
            start = 0
            for i, n in enumerate(clients_per_tier):
                tiered_clients[i+1] = clients[start:start + n]
                start += n
            print(tier_method)
        return tiered_clients