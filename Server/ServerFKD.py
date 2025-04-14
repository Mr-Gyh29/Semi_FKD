
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy, soft_predict
from Server.ServerBase import Server
from Client.ClientFKD import ClientFKD
from tqdm import tqdm
import numpy as np
from utils import FedAvg
from mem_utils import MemReporter
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc
import math
import torch.nn.functional as F

class ServerFKD(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test, pub_test,logger,device,duration_round):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)
        dict_pub = [np.random.randint(low=0,high=10000,size = 1000)] 
        # Set the public data
        self.public_data = LocalDataloaders(pub_test,dict_pub,args.batch_size,ShuffleorNot = False,frac=1)[0]
        self.duration_round = duration_round
        self.lr_tierdict = {} # last round tier dict


    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFKD(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], loader_pub = self.public_data, idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device, h=self.h[idx]))
    
    def Compute_traininglatency(self, idle_clients):
        latency_list = []
        energy_list = []
        for client in idle_clients:
            # 计算每个用户的训练时延
            latency, energy = self.LocalModels[client].Compute_latency_and_energy()
            latency_list.append(latency)
            energy_list.append(energy)
        return latency_list, energy_list
    
    # 空闲用户识别，这一步应该在回合开始时进行
    def Indentify_idelclients(self, epoch):
        idle_clients = []
        # 先判断当前属于第几个回合，以及上个回合的分层情况，
        if epoch == 1:
            return [i for i in range(self.args.num_clients)]
        else:
            for i in range(2, epoch+1):
                idle_clients.extend(self.lr_tierdict['{}'.format(i)])
            return idle_clients
    
    # 为空闲用户分层，这一步在对空闲用户分配资源后
    def Tier_clients(self, epoch, idle_clients):   
        # 这里应该先识别空闲用户，空闲用户的层级不变，其他用户根据计算时延进行分层
        client_latency, _ = self.Compute_traininglatency(idle_clients) 
        tier_client = []
        for idx, latency in enumerate(client_latency):
            tier_client.append(math.ceil(latency/self.duration_round))
        tier_dict = {}
        for idx, tier in enumerate(tier_client):
            if tier not in tier_dict:
                tier_dict[tier] = []
            tier_dict[tier].append(idx)
        self.lr_tierdict = tier_dict
    
    def Compute_cosine_similarity(self, client1, client2):
        # 计算两个用户输出的logits的余弦相似度
        logits_1 = self.LocalModels[client1].generate_knowledge(temp=self.args.temp)
        logits_2 = self.LocalModels[client2].generate_knowledge(temp=self.args.temp)
        logits_1 = F.softmax(logits_1, dim=1)
        logits_2 = F.softmax(logits_2, dim=1)
        cosine_similarity = F.cosine_similarity(logits_1, logits_2, dim=1).mean().item()
        return cosine_similarity

    # 匹配相关用户
    def Get_relevatn_clients(self, which_client):
        relevant_clients = []
        cosine_similarity_list = []
        for client in range(self.args.num_clients):
            if client != which_client:
                # 计算余弦相似度
                cosine_similarity = self.Compute_cosine_similarity(which_client, client)
                cosine_similarity_list.append(cosine_similarity)
                # 选择余弦相似度最高的用户
                sorted_indices = sorted(range(len(cosine_similarity_list)), key=lambda k: cosine_similarity_list[k], reverse=True)
                relevant_clients = sorted_indices[:self.args.relevant_clients_num]
        return relevant_clients  

    
    def Get_public_knowledge(self, relevant_clients, temp):
         # record the soft predictions for each client
        Knowledges = []
        global_soft_prediciton = []  # Initialize a list to store global soft predictions
        for client in relevant_clients:
            knowledges = self.LocalModels[client].generate_knowledge(temp=self.args.temp)  # Generate knowledge from the local model
            Knowledges.append(torch.stack(knowledges))  # Append the knowledge to the Knowledges list
        
        batch_pub = Knowledges[0].shape[0]  # Get the batch size of the public data
        for i in range(batch_pub):  # Iterate over the batch size
            num = Knowledges[0].shape[1]  # Get the number of classes, the sample number of each batch
            soft_label = torch.zeros(num, self.args.num_classes)  # Initialize a tensor to store soft labels
            for idx in relevant_clients:  # Iterate over the selected clients
                soft_label += Knowledges[idx][i]  # Accumulate the soft labels from the clients
            soft_label = soft_label / len(relevant_clients)  # Average the soft labels, the demision of the soft label is [batch_size, num_classes]
            global_soft_prediciton.append(soft_label)  # Append the soft label to the global soft predictions list, the demision of the global soft prediction is [batch_num, batch_size, num_classes]
        del Knowledges  # Delete the Knowledges list to free memory
        return global_soft_prediciton
    

    def train(self):
        # 整个训练过程一共有多个训练回合，每个回合内的流程如下：
        # 1. 回合开始时，识别当前回合的空闲用户
        # 2. 对空闲用户分配剩余可用资源
        # 3. 计算每个用户的训练时延
        # 4. 对空闲用户进行分层
        # 5. 为每个空闲用户选择的相关用户，平均相关用户的知识，获得该空闲用户的集成soft prediction
        # 6. 使用每个空闲用户的集成soft prediction，进行本地知识蒸馏训练
        # 7. 进入下一回合，进行新的一轮训练流程迭代
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        for epoch in range(self.args.num_rounds):
            test_accuracy = 0
            local_losses = []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idle_clients = self.Indentify_idelclients(epoch)
            self.Allocation_resource(idle_clients)
            # 返回每个用户在哪一层，或者每一层有哪些用户？
            # c
            self.Tier_clients(epoch, idle_clients)
            for client in idle_clients:
                # 选择相关用户
                relevant_clients = self.Get_relevatn_clients(client)
                # 获取相关用户的知识
                global_soft_prediciton = self.Get_knowledge_per_client(relevant_clients, self.args.temp)
                # 进行本地知识蒸馏训练
                w, loss = self.LocalModels[client].Local_train(global_soft_prediciton, self.args.lam, self.args.temp, epoch)
                local_losses.append(copy.deepcopy(loss))
                acc = self.LocalModels[client].test_accuracy()
                test_accuracy += acc
            gc.collect()
            # 更新全局模型
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average local test accuracy:', test_accuracy / self.args.num_clients)
            print('global test accuracy: ', self.global_test_accuracy())
        
        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()
    
    