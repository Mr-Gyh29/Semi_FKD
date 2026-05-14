
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy, soft_predict
from Server.ServerBase import Server
from Client.ClientFKD import ClientFKD
# from tqdm import tqdm
import numpy as np
from utils import FedAvg
from mem_utils import MemReporter
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc
import math
import torch.nn.functional as F
from PSO import PSO
import cvxpy as cp
import random
import ecos


class ServerFKD(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test, pub_test,logger, device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)
        dict_pub = [np.random.randint(low=0,high=10000,size = 1000)] 
        # Set the public data
        self.public_data = LocalDataloaders(pub_test,dict_pub,args.batch_size,ShuffleorNot = False,frac=1)[0]
        self.duration_round = args.duration_round
        self.lr_tierdict = {} # last round tier dict


    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFKD(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], self.global_testloader, loader_pub = self.public_data, idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device, h=self.h[idx]))
    
    def Compute_traininglatency(self, idle_clients, ratio_B, ratio_F):
        latency_list = [0.00 for _ in range(self.args.num_clients)]
        energy_list = [0.00 for _ in range(self.args.num_clients)]
        for client in idle_clients:
            # 计算每个用户的训练时延
            latency, energy = self.LocalModels[client].Compute_latency_and_energy(ratio_B[client], ratio_F[client])
            latency_list[client] = latency
            energy_list[client] = energy
        return latency_list, energy_list
    
    # 空闲用户识别，这一步应该在回合开始时进行
    def Indentify_idelclients(self, epoch):
        idle_clients = [client for i, clients in self.lr_tierdict.items() if epoch % i == 0 for client in clients]
        a = np.zeros(self.args.num_clients, dtype=int)
        if idle_clients:
            a[np.array(idle_clients, dtype=int)] = 1
        return a
    
    
    # # 为空闲用户分层，这一步在对空闲用户分配资源后
    # def Tier_clients(self, epoch, idle_clients):   
    #     # 这里应该先识别空闲用户，空闲用户的层级不变，其他用户根据计算时延进行分层
    #     client_latency, _ = self.Compute_traininglatency(idle_clients)
    #     print("client_latency: ", client_latency)
    #     tier_client = [] # 记录每个用户在第几层
    #     for idx, latency in enumerate(client_latency):
    #         tier_client.append(math.ceil(latency/self.duration_round))
    #     print("tier_client: ", tier_client)
    #     tier_dict = copy.deepcopy(self.lr_tierdict) # 记录每个用户在第几层
    #     for idx, tier in enumerate(tier_client):
    #         if tier != 0:
    #             if tier not in tier_dict:
    #                 tier_dict[tier] = []
    #             tier_dict[tier].append(idx)
    #     self.lr_tierdict = copy.deepcopy(tier_dict)
    #     del tier_dict

    # def Tier_clients(self, idle_clients, ratio_B, ratio_F):
    #     # 为空闲用户分层，这一步在对空闲用户分配资源后
    #     # 计算空闲用户的时延
    #     client_latency, _ = self.Compute_traininglatency(idle_clients, ratio_B, ratio_F)
    #     print("client_latency: ", client_latency)
        
    #     # 初始化分层字典
    #     tier_dict = copy.deepcopy(self.lr_tierdict)

    #     # 遍历空闲用户，更新分层信息
    #     for client in idle_clients:
    #         # 根据时延计算该用户的新层级
    #         new_tier = math.ceil(client_latency[client] / self.duration_round)

    #         # 从旧层级中移除该用户
    #         for tier, clients in tier_dict.items():
    #             if client in clients:
    #                 clients.remove(client)

    #         # 将用户加入新层级
    #         if new_tier != 0:
    #             if new_tier not in tier_dict:
    #                 tier_dict[new_tier] = []
    #             tier_dict[new_tier].append(client)

    #     # 更新全局分层字典
    #     self.lr_tierdict = copy.deepcopy(tier_dict)
    #     del tier_dict

    def Get_active_clients(self, ratio_B, ratio_F, a):
        ac_clients = np.zeros(self.args.num_clients, dtype=int)
        for g_client in range(self.args.num_clients):
            client_E = self.LocalModels[g_client].compute_energy(ratio_B[g_client], ratio_F[g_client])
            client_T = self.LocalModels[g_client].compute_latency(ratio_B[g_client], ratio_F[g_client])
            tau = next((tier for tier, clients in self.lr_tierdict.items() if g_client in clients), 0)
            if tau == 0:
                raise ValueError(f"Client {g_client} is not in any tier!!!!!!!!!!!!!!!!!!!")
            # 判断用户是否满足能耗和时延要求
            if client_E < self.args.energy_threshold and client_T < a[g_client] * tau * self.duration_round:
                ac_clients[g_client] = 1
        return ac_clients

    
    # 出现nan
    # 余弦相似度作为知识聚合指标
    def Compute_cosine_similarity(self, client1, client2):
        # 计算两个用户输出的logits的余弦相似度
        logits_1 = self.LocalModels[client1].generate_knowledge(temp=self.args.temp)
        logits_2 = self.LocalModels[client2].generate_knowledge(temp=self.args.temp)
        # print((len(logits_1)==len(logits_2)))
        accum_cosine_simi = 0
        for i in range(len(logits_1)):
            logit_1 = F.softmax(logits_1[i], dim=1)
            logit_2 = F.softmax(logits_2[i], dim=1)
            # if torch.isnan(logit_1).any()  or torch.isinf(logit_1).any():
            #     print("NaN or Inf detected in logits!")
            #     print("logit_1: ", logits_1[i])
            # if torch.isnan(logit_2).any()  or torch.isinf(logit_2).any():
            #     print("NaN or Inf detected in logits!")
            #     print("logit_2: ", logits_2[i])         
            cosine_similarity = F.cosine_similarity(logit_1, logit_2, dim=1).mean().item()
            accum_cosine_simi += cosine_similarity
        return accum_cosine_simi

    # 获得权重
    def Get_teacher_weights(self, which_client, teacher_clients):
        cosine_similarity_list = []
        for client in teacher_clients:
            if client != which_client:
                # 计算余弦相似度
                cosine_similarity = self.Compute_cosine_similarity(which_client, client)
                cosine_similarity_list.append(cosine_similarity)
            else:
                continue
        # print("client {}'s cosine similarity list is {}".format(which_client, cosine_similarity_list))    
        teacher_weights = [cosine_similarity_list[i]/ sum(cosine_similarity_list) for i in range(len(cosine_similarity_list))]
        teacher_weights = np.ones(len(teacher_clients))
        return teacher_weights
    
    def Get_knowledge_per_client(self, which_client, teacher_clients, temp):
        if teacher_clients is None or len(teacher_clients) == 0:
            teacher_weights = [0.0 for _ in range(self.args.num_clients)]
            teacher_nums = 1
        else:
            teacher_weights = self.Get_teacher_weights(which_client, teacher_clients)
            teacher_nums = len(teacher_clients)
         # record the soft predictions for each client
        Knowledges = []
        global_soft_prediciton = []  # Initialize a list to store global soft predictions
        for client in range(self.args.num_clients):
            knowledges = self.LocalModels[client].generate_knowledge(temp=self.args.temp)  # Generate knowledge from the local model
            Knowledges.append(torch.stack(knowledges))  # Append the knowledge to the Knowledges list
        
        batch_pub = Knowledges[0].shape[0]  # Get the batch size of the public data
        # print("batch_pub: ", batch_pub)
        # print("len(Knowledges)", len(Knowledges))
        # print("Knowledges[0].shape", Knowledges[0].shape)
        for i in range(batch_pub):  # Iterate over the batch size
            num = Knowledges[0].shape[1]  # Get the number of classes, the sample number of each batch
            soft_label = torch.zeros(num, self.args.num_classes)  # Initialize a tensor to store soft labels
            # if teacher_nums == 1:
            #     soft_label += Knowledges[which_client][i]
            # else:

            for idx, idx_client in enumerate(teacher_clients):  # Iterate over the selected clients
                # print("idx: ", idx)
                # print("i", i)
                soft_label += teacher_weights[idx]*Knowledges[idx_client][i]  # Accumulate the soft labels from the clients
            soft_label = soft_label / teacher_nums  # Average the soft labels, the demision of the soft label is [batch_size, num_classes]
            # print("!!!!!!!!!!!看这里!!!!!!!!!!!!!!")
            # print(soft_label)
            global_soft_prediciton.append(soft_label)  # Append the soft label to the global soft predictions list, the demision of the global soft prediction is [batch_num, batch_size, num_classes]
        del Knowledges  # Delete the Knowledges list to free memory
        return global_soft_prediciton

    def compute_xi(self, client, s_k):
        # print("Computing xi for client {}, s_k: {}".format(client, s_k))
        # 这里要重新调用get_knowledge_per_client来获取教师的logits,并按batch处理
        if s_k[client].sum() != 0:
            # 如果没有教师，则返回0          
            sp = self.Get_knowledge_per_client(client, [c for c in range(self.args.num_clients) if s_k[client][c] == 1], self.args.temp)
            # print("sp", sp)
            sp = torch.stack(sp)
        loss_list = []
        model = self.LocalModels[client].model
        # 检查sp是否有NaN或Inf
        if s_k[client].sum() != 0:
            if torch.isnan(sp).any() or torch.isinf(sp).any():
                print("NaN or Inf detected in global_soft_prediction!")
                return
        for batch_idx, (X, y) in enumerate(self.LocalModels[client].trainloader):              
            X = X.to(self.LocalModels[client].device)
            y = y.to(self.LocalModels[client].device)
            _,Z = model(X)
            if torch.isnan(Z).any() or torch.isinf(Z).any():
                print("NaN or Inf detected in Z!")
                print("Z:", Z)
                return
            if self.args.loss_F == "CEKD":
                # loss1 = self.LocalModels[client].ce(Z,y)
                loss1 = F.cross_entropy(Z, y)
            else:
                loss1 = F.mse_loss(Z, F.one_hot(y, num_classes=self.args.num_classes).float())
            loss2 = torch.tensor(0.0).to(self.LocalModels[client].device)

            if s_k[client].sum() != 0:
                for idx, (X_pub,y_pub) in enumerate(self.LocalModels[client].loader_pub):
                    # 检查
                    if idx == batch_idx:
                        X_pub = X_pub.to(self.LocalModels[client].device)
                        y_pub = y_pub.to(self.LocalModels[client].device)
                        _,Z_pub = model(X_pub)
                        Q_pub = soft_predict(Z_pub, self.args.temp).to(self.LocalModels[client].device)
                                # 检查输入是否有 NaN 或 Inf
                        if torch.isnan(Q_pub).any() or torch.isinf(Q_pub).any():
                            print("NaN or Inf detected in Q_pub!")
                            return

                        # 检查并修正 Q_pub 和 sp[idx] 的 NaN 或 Inf
                        if torch.isnan(Q_pub).any() or torch.isinf(Q_pub).any():
                            Q_pub = torch.where(torch.isnan(Q_pub) | torch.isinf(Q_pub), torch.zeros_like(Q_pub), Q_pub)
                        sp_idx = sp[idx].to(self.LocalModels[client].device)
                        if torch.isnan(sp_idx).any() or torch.isinf(sp_idx).any():
                            sp_idx = torch.where(torch.isnan(sp_idx) | torch.isinf(sp_idx), torch.zeros_like(sp_idx), sp_idx)
                        if s_k[client].sum() != 0:
                            if self.args.loss_F == "CEKD":
                                loss2 -= F.kl_div(Q_pub, sp_idx, reduction='batchmean')
                            else:
                                # print("正在进行MSE-loss2")
                                loss2 += F.mse_loss(Q_pub, sp_idx)
            loss = loss1 + self.args.lam * loss2
            loss_norm = torch.norm(loss, p=2) ** 2
            loss_list.append(loss_norm.item())
        loss_sum = sum(loss_list)
        local_loader = self.Loaders_train[client]
        num_samples = len(local_loader.dataset)
                    

        # 求平均
        xi_k = loss_sum * (1 / num_samples)

        return xi_k

    def compute_subprone(self, all_clients):
        # self.Tier_clients(all_clients, ratio_B, ratio_F)
        s_k = np.ones((self.args.num_clients, self.args.num_clients), dtype=int)
        # 使s_k中的对角线元素为0
        np.fill_diagonal(s_k, 0)
        for client in all_clients:
            max_xi_k = self.compute_xi(client, s_k)
            # 利用贪心思想，随机将s_k[client]中值为1的元素改为0，判断xi_k是否增加，如果增加，则继续改为0，然后再将s_k[client]中另一个值为1的元素改为0，继续判断，直到xi_k不再增加
            # s_k_copy = copy.deepcopy(s_k)
            for idx in range(len(s_k[client])):
                if s_k[client][idx] == 1:
                    s_k[client][idx] = 0
                    xi_k_new = self.compute_xi(client, s_k)
                    # print("max_xi_k", max_xi_k)
                    # print(f"xi_k_new for client {client} with idx {idx}: {xi_k_new}")
                    if xi_k_new > max_xi_k:
                        max_xi_k = xi_k_new
                        # s_k = copy.deepcopy(s_k_copy)
                        s_k[client][idx] = 0
                    else:
                        s_k[client][idx] = 1

        # # 这里我想用PSO来优化教师选择，
        # s_k = np.ones((self.args.num_clients, self.args.num_clients), dtype=int)
        # np.fill_diagonal(s_k, 0)
        # mypso = PSO(pN=30, dim=self.args.num_clients, max_iter=50)
        # mypso.init_Population()
        # for t in range(mypso.max_iter):
        #     for i in range(mypso.pN):  # 更新gbest\pbest
        #         temp_s_k = np.zeros((self.args.num_clients, self.args.num_clients), dtype=int)
        #         for client in all_clients:
        #             for j in range(mypso.dim):
        #                 if j != client:
        #                     if mypso.X[i][j] >= 0.5:
        #                         temp_s_k[client][j] = 1
        #                     else:
        #                         temp_s_k[client][j] = 0
        #                 else:
        #                     temp_s_k[client][j] = 0
        #         temp = 0
        #         for client in all_clients:
        #             xi_k = self.compute_xi(client, temp_s_k)
        #             temp += xi_k
        #         if temp > mypso.p_fit[i]:  # 更新个体最优
        #             mypso.p_fit[i] = temp
        #             mypso.pbest[i] = mypso.X[i]
        #             if mypso.p_fit[i] > mypso.fit:  # 更新全局最优
        #                 mypso.gbest = mypso.X[i]
        #                 mypso.fit = mypso.p_fit[i]
        #         print(f"Iteration {t}, Particle {i}, Fitness: {mypso.p_fit[i]:.4f}, Best Fitness: {mypso.fit:.4f}")
        #     for i in range(mypso.pN):
        #         mypso.V[i] = mypso.w * mypso.V[i] + mypso.c1 * mypso.r1 * (mypso.pbest[i] - mypso.X[i]) + mypso.c2 * mypso.r2 * (mypso.gbest - mypso.X[i])
        #         mypso.X[i] = mypso.X[i] + mypso.V[i]
        #         mypso.X[i] = np.clip(mypso.X[i], 0.0, 1.0)
        # # 根据mypso.gbest来生成最终的s_k
        # s_k = np.zeros((self.args.num_clients, self.args.num_clients), dtype=int)
        # for client in all_clients:
        #     for j in range(mypso.dim):
        #         if j != client:
        #             if mypso.gbest[j] >= 0.5:
        #                 s_k[client][j] = 1
        #             else:
        #                 s_k[client][j] = 0
        #         else:
        #             s_k[client][j] = 0
                
        return s_k
    
    def compute_subprtwo(self, s_k, a):
        # 空闲用户的数量一直在改变
        mypso = PSO(pN=30, dim=2*self.args.num_clients, max_iter=100)
        mypso.init_Population()
        best_pN_ac = np.zeros(self.args.num_clients, dtype=int)
        lambda_penalty = 0.05
        for t in range(mypso.max_iter):
            for i in range(mypso.pN):  # 更新gbest\pbest
                ratio_B, ratio_F = mypso.X[i][:self.args.num_clients], mypso.X[i][self.args.num_clients:]
                active_client_list = self.Get_active_clients(ratio_B, ratio_F, a)
                temp = 0
                # 应该计算所有用户的xi和
                for ix, is_active in enumerate(active_client_list):
                    if is_active:
                        # 计算xi_k
                        xi_k = self.compute_xi(ix, s_k)
                        temp += xi_k
                # 计算ratio_B的罚函数
                total_ratio_B = np.sum(ratio_B)
                penalty_ratio_B = np.abs(1 - total_ratio_B) + np.sum(np.maximum(0, -ratio_B)) + np.sum(np.maximum(0, ratio_B - 1))
                # 计算ratio_F的罚函数
                penalty_ratio_F = np.sum(np.maximum(0, -ratio_F)) + np.sum(np.maximum(0, ratio_F - 1)) 

                # 计算时延的罚函数
                # 计算能耗的罚函数
                penalty_energy = 0
                penalty_latency = 0
                for idc in range(self.args.num_clients):
                    client_e = self.LocalModels[idc].compute_energy(ratio_B[idc], ratio_F[idc])
                    client_t = self.LocalModels[idc].compute_latency(ratio_B[idc], ratio_F[idc])
                    tau = next((tier for tier, clients in self.lr_tierdict.items() if idc in clients), 0)
                    if tau == 0:
                        raise ValueError(f"Client {idc} is not in any tier!!!!!!!!!!!!!!!!!!!")
                    penalty_energy_client = max(0,active_client_list[idc] * client_e - self.args.energy_threshold)
                    penalty_latency_client = max(0, active_client_list[idc] * client_t - a[idc] * tau * self.duration_round)
                    penalty_energy += penalty_energy_client
                    penalty_latency += penalty_latency_client
                total_penalty = penalty_ratio_B + penalty_ratio_F + penalty_energy + penalty_latency
                # 计算目标函数
                temp = temp - lambda_penalty * total_penalty
                
                if temp > mypso.p_fit[i]:  # 更新个体最优
                    mypso.p_fit[i] = temp
                    mypso.pbest[i] = mypso.X[i]
                    if mypso.p_fit[i] > mypso.fit:  # 更新全局最优
                        mypso.gbest = mypso.X[i]
                        mypso.fit = mypso.p_fit[i]
                        best_pN_ac = copy.deepcopy(active_client_list)
                print(f"Iteration {t}, Particle {i}, Fitness: {mypso.p_fit[i]:.4f}, Best Fitness: {mypso.fit:.4f}")
                # print(" ")
            for i in range(mypso.pN):
                mypso.V[i] = mypso.w * mypso.V[i] + mypso.c1 * mypso.r1 * (mypso.pbest[i] - mypso.X[i]) + \
                            mypso.c2 * mypso.r2 * (mypso.gbest - mypso.X[i])
                mypso.X[i] = mypso.X[i] + mypso.V[i]
                mypso.X[i] = np.clip(mypso.X[i], 0.1, 1)
        return best_pN_ac, mypso.X[:self.args.num_clients], mypso.X[self.args.num_clients:]
    
    
    def compute_subprthree(self, s_k, a):
        b_prev = np.ones(self.args.num_clients)/ self.args.num_clients
        b_prev = np.clip(b_prev, 1e-3, 1.0)
        xi_k = np.array([self.compute_xi(c, s_k[c]) for c in range(self.args.num_clients)])
        # xi_k = [7.21282627e-06, 4.81088582e-09, 1.33635035e-08, 2.39286035e-03,
        #          2.55246448e-06, 3.06278146e-07, 4.01408817e-06, 1.37816648e-07,
        #          3.17230473e-06, 1e-12]
        # xi_k = xi_k/sum(xi_k)
        b_best = b_prev.copy()
        prob_best = -99999999
        for it in range(self.args.CVX_iterations):
            ratio_B = cp.Variable(self.args.num_clients, nonneg=True)
            ratio_F = cp.Variable(self.args.num_clients, nonneg=True)
            b = cp.Variable(self.args.num_clients, nonneg=True)
            ratio_B.value = np.ones(self.args.num_clients) / self.args.num_clients
            ratio_F.value = np.ones(self.args.num_clients) * 0.5
            b.value = b_prev.copy()
            T_k  = np.array([self.LocalModels[c].compute_latency_cvx(ratio_B[c], ratio_F[c]) for c in range(self.args.num_clients)])
            E_k  = np.array([self.LocalModels[c].compute_energy_cvx(ratio_B[c], ratio_F[c]) for c in range(self.args.num_clients)])
            # 添加约束条件
            constraints = []
            constraints.append(cp.sum(ratio_B) == 1)  # 带宽比例之和为1
            constraints += [ratio_B >= 0, ratio_B <= 1]
            constraints += [ratio_F >= 0, ratio_F <= 1]
            constraints += [b >= 1e-3, b <= 1]
            for c in range(self.args.num_clients):
                tau = next((tier for tier, clients in self.lr_tierdict.items() if c in clients), 0)
                inv_b_prev = 1.0 / b_prev[c]
                inv_b_prev_sq = 1.0 / (b_prev[c] ** 2)
                taylor = inv_b_prev + inv_b_prev - b[c] * inv_b_prev_sq
                constraints.append(a[c] * tau * self.args.duration_round * taylor >= T_k[c])
                constraints.append(self.args.energy_threshold * taylor >= E_k[c])
            print("xi_k", xi_k)


            xi_k = np.array(xi_k)
            max_val = np.max(xi_k)
            max_exp = np.floor(np.log10(max_val))
            print("max_exp:", max_exp)
            # 允许的最小指数级
            min_exp = max_exp - 6
            # 找出指数级低于min_exp的元素，设为0
            mask = np.log10(xi_k, where=xi_k>0) < min_exp
            xi_k[mask] = 1e-10
            # 自动调整 coe，使其与 xi_k 的指数级相反
            xi_k_magnitude = np.min(np.abs(xi_k))
            if xi_k_magnitude > 0:
                coe = 10 ** (-int(np.floor(np.log10(xi_k_magnitude))))
            else:
                coe = 1.0
            print("coe:", coe)
            print("Regulation", (1.0 - 2.0 * b_prev))
            coeff = coe * xi_k - self.args.lambda2 * (1.0 - 2.0 * b_prev)
            obj = cp.Maximize(coeff @ b)
            # DCP check
            if not obj.is_dcp():
                print("[DCP CHECK] Objective function is not DCP!")
                print(obj)
                raise ValueError("DCP violation in objective.")
            for i, con in enumerate(constraints):
                if not con.is_dcp():
                    print(f"[DCP CHECK] Constraint #{i} is not DCP!")
                    print(con)
                    raise ValueError(f"DCP violation in constraint #{i}")
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
            # try:
            #     prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            #     if prob.status not in ["optimal", "optimal_inaccurate"]:
            #         raise ValueError(f"ECOS failed with status {prob.status}")
            # except Exception as e:
            #     print(f"[WARN] ECOS failed: {e}")
            #     print("[INFO] Switching to SCS fallback solver...")
            #     try:
            #         prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
            #     except Exception as e2:
            #         print(f"[ERROR] SCS also failed: {e2}")
            #         return None, None, None
            print(f"Solver status: {prob.status}")
            b_new = b.value
            print("b_new:", b_new)
            if b_new is None:
                print("CVXPY failed to solve the problem!")
                break
            # 取迭代次数内目标值最大的那次
            if prob.value > prob_best:
                prob_best = prob.value
                b_best = b_new
            diff  = np.linalg.norm(b_new - b_prev, ord=2)
            print(f"Iter {it}: obj={prob.value:.4f}, best_obj={prob_best:.4f}, ||b_new-b_prev||₂={diff:.6f}")
            if diff < self.args.tol:
                print("Converged.")
                break
            b_prev = np.clip(b_new, 1e-3, 1.0)
        print("Final b:", b_best)
        return b_best, ratio_B.value, ratio_F.value
    
    def Allocation_resource(self, epoch, all_clients):
        # 根据epoch和self.lr_tierdict来识别空闲用户,如果空闲那么a[client] = 1，否则a[client] = 0
        a = self.Indentify_idelclients(epoch)
        method_Bandwidth = self.args.method_Bandwidth
        method_Frequency = self.args.method_Frequency
        method_teacher = self.args.method_Teacher

        if method_teacher == 'Random':
            s_k = np.zeros((self.args.num_clients, self.args.num_clients), dtype=int)
            # 随机使s_k[i]中的n个值为1，n也是一个随机数
            for client in all_clients:
                num_teachers = np.random.randint(1, self.args.num_clients)
                # 随机选择num_teachers个教师
                teacher_indices = np.random.choice([c for c in all_clients if c != client], size=num_teachers, replace=False)
                s_k[client, teacher_indices] = 1
        
        elif method_teacher == 'Cosine':
            s_k = np.zeros((self.args.num_clients, self.args.num_clients), dtype=int)
            # 计算每个用户的余弦相似度
            for client in all_clients:
                # 计算余弦相似度
                cosine_similarities = [self.Compute_cosine_similarity(client, other_client) if other_client != client else 0 for other_client in all_clients]
                # 根据余弦相似度选择余弦相似度最大的作为教师（排除自身）
                cos_arr = np.array(cosine_similarities, dtype=float)
                cos_arr[client] = -np.inf  # 排除自身
                best_idx = int(np.argmax(cos_arr))
                s_k[client, best_idx] = 1

        elif method_teacher == 'IL':
            s_k = np.zeros((self.args.num_clients, self.args.num_clients), dtype=int)

        elif method_teacher == 'Greedy':
            s_k = np.ones((self.args.num_clients, self.args.num_clients), dtype=int)
            np.fill_diagonal(s_k, 0)
            # s_k = self.compute_subprone(all_clients)

        elif method_teacher == 'JSTSP':
            s_k = 0


        if method_Bandwidth == 'PSO' and method_Frequency == 'PSO':
            b, r_B, r_F = self.compute_subprtwo(s_k, a)
 
        elif method_Bandwidth == 'CVX' and method_Frequency == 'CVX':
            b, r_B, r_F = self.compute_subprthree(s_k, a)
            print(type(b))
            print("b_float:", b)
            b = np.round(np.array(b)).astype(int)
            print("b:", b)
        else:
            if method_Bandwidth == 'Equal':
                r_B = [1.0 / self.args.num_clients] * self.args.num_clients

            # 带宽高的分配少
            if method_Bandwidth == 'Proportional':
                h_abs2 = np.abs(np.array(self.h)) ** 2
                total_gain = np.sum(h_abs2)
                r_B = (1 - (h_abs2 / total_gain)).tolist()
            elif method_Bandwidth == 'Random':
                r_B = np.random.dirichlet(np.ones(self.args.num_clients))

            if method_Frequency == 'Equal':
                r_F = [0.5 for _ in range(self.args.num_clients)]

            elif method_Frequency == 'Maximum':
                r_F = [1.0 for _ in range(self.args.num_clients)]
            
            elif method_Frequency == 'Random':
                r_F = np.random.uniform(low=1e-8, high=1.0, size=self.args.num_clients)
            b = self.Get_active_clients(r_B, r_F, a)  # 获取满足能耗和时延要求的用户

        return s_k, b


    

    

    def train(self):
        # 首先对所用用户分层，分层情况并不跟随训练回合改变而改变
        # 整个训练过程一共有多个训练回合，每个回合内的流程如下：
        # 对当前回合对应的空闲用户进行一次资源分配策略优化
        # 根据给出的资源分配策略，计算每个用户的时延和能耗阈值选择可用用户
        # 为可用用户选择教师，并以余弦相似度为权重聚合教师知识
        # 每个可用用户进行本地训练
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        loss1_epoch = []
        local_acc_epoch = []
        global_acc_epoch = []
        all_clients = [i for i in range(self.args.num_clients)]
        self.lr_tierdict = self.Tier_clients(self.args.method_tier, self.args.num_tiers)  # 初始化分层字典
        # 初始化本地模型
        for epoch in range(1, self.args.num_epochs+1):
            # 这里应该加入h的变化
            # 每次epoch选择N个用户
            test_accuracy = 0
            test_global_accuracy = 0
            local_losses = []
            local_loss1 = []
            print(f'\n | Global Training Round : {epoch} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            # 进行优化，获得每回合用户激活指示alpha，教师选择指示s_k，带宽分配比例ratio_B，计算能力分配比例ratio_F
            if self.args.is_allocation:
                s_k, alpha = self.Allocation_resource(epoch, all_clients)
            else:
                s_k = np.zeros((self.args.num_clients, self.args.num_clients), dtype=int)
                alpha = np.ones(self.args.num_clients, dtype=int)  # 所有用户都激活
            # alpha记录了当前回合选择激活哪些用户进行FL
            active_clients = [i for i in range(self.args.num_clients) if alpha[i] == 1]
            print("active_clients: ", active_clients)
            for i in active_clients:
                print("Client {0}'s s_k are:{1}".format(i, s_k[i]))
            # 如果没有空闲用户，则直接进行本地训练
            if not active_clients:
                print("No active clients in this round.")
                continue
            for client in active_clients:
                # 根据s_k，为每个用户选择相应的教师
                teacher_clients = [c for c in range(self.args.num_clients) if s_k[client][c] == 1]
                # 根据余弦相似度计算权重，聚合知识
                global_soft_prediciton = self.Get_knowledge_per_client(client, teacher_clients, self.args.temp)
                # 进行本地知识蒸馏训练
                w, loss, loss1 = self.LocalModels[client].Local_train(global_soft_prediciton, self.args.lam, self.args.temp, epoch)

            for c in range(self.args.num_clients):
                bcloss1_list = []
                for batch_idx, (X, y) in enumerate(self.LocalModels[c].trainloader):
                    bcloss1_temp = self.LocalModels[c].compute_loss1(X, y)
                    bcloss1_list.append(bcloss1_temp.item())
                losss1 = sum(bcloss1_list) / len(bcloss1_list)
                local_losses.append(losss1)
                acc = self.LocalModels[c].test_accuracy()
                global_acc = self.LocalModels[c].test_global_accuracy()
                test_accuracy += acc
                test_global_accuracy += global_acc
            gc.collect()
            # 更新全局模型
            # loss_avg = sum(local_losses) / len(local_losses)
            loss1_avg = sum(local_losses) / len(local_losses)
            acc_avg = test_accuracy / len(local_losses)
            acc_global_avg = test_global_accuracy / len(local_losses)
            loss1_epoch.append(loss1_avg)
            local_acc_epoch.append(acc_avg)
            global_acc_epoch.append(acc_global_avg)
            # train_loss.append(loss_avg)
            self.logger.add_scalar('loss', loss1_avg, epoch)
            self.logger.add_scalar('accuracy', acc_avg, epoch)
            # self.logger.close()
            # print("len(local_losses)", len(local_losses))
            print("average loss:  ", loss1_avg)
            print('average local test accuracy:', acc_avg)
            print('average global test accuracy: ', acc_global_avg)

        # 绘制loss1_epoch, local_acc_epoch, global_acc_epoch曲线，横坐标都为Epoch，纵坐标分别为Loss1, Average Local Accuracy, Average Global Accuracy
        self.plot_figures(loss1_epoch, local_acc_epoch, global_acc_epoch)
        # 将数据保存成文件
        result_path = './results'
        np.savetxt(result_path + '/loss1_epoch_{}_{}_{}_{}_{}Tier_lam{}_non{}.txt'.format(self.args.method_Teacher, self.args.sample_method, self.args.loss_F, self.args.method_Bandwidth, self.args.num_tiers, self.args.lam, self.args.beta), np.array(loss1_epoch))
        np.savetxt(result_path + '/local_acc_epoch_{}_{}_{}_{}_{}Tier_lam{}_non{}.txt'.format(self.args.method_Teacher, self.args.sample_method, self.args.loss_F, self.args.method_Bandwidth, self.args.num_tiers, self.args.lam, self.args.beta), np.array(local_acc_epoch))
        np.savetxt(result_path + '/global_acc_epoch_{}_{}_{}_{}_{}Tier_lam{}_non{}.txt'.format(self.args.method_Teacher, self.args.sample_method, self.args.loss_F, self.args.method_Bandwidth, self.args.num_tiers, self.args.lam, self.args.beta), np.array(global_acc_epoch))
        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()


    
