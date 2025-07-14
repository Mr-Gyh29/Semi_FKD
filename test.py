# # import numpy as np
# # import torch
# # import time
# # import torch.nn.functional as F
# # # # a = {'a':[1, 2, 3]}
# # # # b = [ ]
# # # # b.extend(a['a'])
# # # # print(b)

# # # # tier_client = [0, 1, 0, 1, 2, 2]
# # # # tier_dict = {}
# # # # for idx, tier in enumerate(tier_client):
# # # #     if tier not in tier_dict:
# # # #         tier_dict[tier] = []
# # # #     tier_dict[tier].append(idx)
# # # # print(tier_dict)

# # # last_tierdict = {'1': [1, 2], '2': [3, 4], '3': [5, 6]}

# # # # current_round = 3
# # # # idle_clients = []
# # # # # 先判断当前属于第几个回合，以及上个回合的分层情况，
# # # # if current_round == 1:
# # # #     print([i for i in range(4)])
# # # # else:
# # # #     for i in range(2, current_round+1):
# # # #         idle_clients.extend(last_tierdict['{}'.format(i)])
# # # #     print(idle_clients)

# # # # a = [[1, 2],[2, 3]]
# # # # print(np.mean(a, axis=0))

# # # # soft_label = torch.zeros(2, 2)
# # # # knowledge = torch.tensor([[1, 2], [3, 4]])
# # # # print(soft_label)
# # # # soft_label += knowledge
# # # # print(soft_label/2)

# # # # input1 = torch.randn(10, 100)
# # # # input2 = torch.randn(10, 100)
# # # # output = F.cosine_similarity(input1, input2)
# # # # print(output.shape)
# # # # print(output)

# # # cosine_similarity_list = [1, 2, 4, 3, 6, 5]
# # # sorted_indices = sorted(range(len(cosine_similarity_list)), key=lambda k: cosine_similarity_list[k], reverse=True)
# # # print(sorted_indices)

# # # def generate_channel_gain(num_users, variances):
# # #     """
# # #     生成多个用户与基站之间的信道增益 h_k。
    
# # #     参数:
# # #     num_users: 用户数量
# # #     variances: 信道增益方差列表，长度应与 num_users 相同
    
# # #     返回:
# # #     h: 信道增益数组，形状为 (num_users,)
# # #     """
# # #     h = np.zeros(num_users, dtype=complex)
# # #     for k in range(num_users):
# # #         real_part = np.random.normal(0, np.sqrt(variances[k]))
# # #         imag_part = np.random.normal(0, np.sqrt(variances[k]))
# # #         h[k] = real_part + 1j * imag_part
# # #     return h

# # # # h = generate_channel_gain(4, [1, 1, 1, 1])
# # # # print(np.abs(h)**2)
# # # a = 1
# # # b =a 
# # # c =a
# # # b+=1
# # # print(c)
# # # print(b)




# # # variance = [np.random.uniform(0, 1) for _ in range(20)]
# # # print(variance)

# # # random_numbers = np.random.uniform(1, 10, 10)
# # # print(random_numbers)
# # import numpy as np
# # tier_dict={1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
# # num = 3  # 你要查找的数字
# # for k, x in tier_dict.items():
# #     print(f"Key: {k}, Value: {x}")



# # same_tier_others = np.array([c for c in next(v for v in tier_dict.values() if num in v) if c != num],dtype=int)
# # a = np.array([1, 2])
# # b = np.array([3, 4])
# # hadamard_product = a * b
# # print(hadamard_product)

# # c = np.zeros((10, 10), dtype=int)
# # print(type(c))


# # print([1]*10)


# # # start_time = time.time()

# # # last_tier_client = {'1':[1,2,3],'2':[4,5,6]}
# # # client = 6
# # # for tier, clients in last_tier_client.items():
# # #     if client in clients:
# # #         # idx = clients.index(client)
# # #         # s_k = clients[:idx] + clients[idx+1:]
# # #         s_k = [c for c in clients if c != client]
# # #         break
# # # print(s_k)  # 输出: [6]

# # # end_time = time.time()
# # # print("运行时间: {:.8f} 秒".format(end_time - start_time))

# # # q_k = np.random.randint(10, 101)
# # # print(q_k)

# # # s_k = np.ones((10, 10), dtype=int)
# # # # s_k[1,1] = 1
# # # client = 1
# # # clients = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# # # s_k[client, s_k[client]==1] = 2
# # # print(s_k[client]==1)
# # # print(s_k[client])

# # # a_list = np.array([], dtype=int)
# # # np.append(a_list, 1)
# # # print(a_list)


# # # sorted_indices = np.argsort(clients)[::-1]  # 降序排序
# # # print(sorted_indices)

# # # coding: utf-8
# # import numpy as np
# # import random
# # import matplotlib.pyplot as plt


# # # ----------------------PSO参数设置---------------------------------
# # class PSO():
# #     def __init__(self, pN, dim, max_iter):
# #         self.w = 0.8
# #         self.c1 = 2
# #         self.c2 = 2
# #         self.r1 = 0.6
# #         self.r2 = 0.3
# #         self.pN = pN  # 粒子数量
# #         self.dim = dim  # 搜索维度
# #         self.max_iter = max_iter  # 迭代次数
# #         self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
# #         self.V = np.zeros((self.pN, self.dim))
# #         self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
# #         self.gbest = np.zeros((1, self.dim))
# #         self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
# #         self.fit = 1e10  # 全局最佳适应值

# #     # ---------------------目标函数-----------------------------
# #     def function(self, X):
# #         return float(X**2-4*X+3)

# #     # ---------------------初始化种群----------------------------------
# #     def init_Population(self):
# #         for i in range(self.pN):
# #             for j in range(self.dim):
# #                 self.X[i][j] = random.uniform(0, 1)
# #                 self.V[i][j] = random.uniform(0, 1)
# #             self.pbest[i] = self.X[i]
# #             tmp = self.function(self.X[i])
# #             self.p_fit[i] = tmp
# #             if tmp < self.fit:
# #                 self.fit = tmp
# #                 self.gbest = self.X[i]

# #                 # ----------------------更新粒子位置----------------------------------

# #     def iterator(self):
# #         fitness = []
# #         for t in range(self.max_iter):
# #             for i in range(self.pN):  # 更新gbest\pbest
# #                 temp = self.function(self.X[i])
# #                 print("X[i]", self.X[i][0])
# #                 if temp < self.p_fit[i]:  # 更新个体最优
# #                     self.p_fit[i] = temp
# #                     self.pbest[i] = self.X[i]
# #                     if self.p_fit[i] < self.fit:  # 更新全局最优
# #                         self.gbest = self.X[i]
# #                         self.fit = self.p_fit[i]
# #             for i in range(self.pN):
# #                 self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
# #                             self.c2 * self.r2 * (self.gbest - self.X[i])
# #                 self.X[i] = self.X[i] + self.V[i]

# #             fitness.append(self.fit)
# #             print(self.X[0], end=" ")
# #             print(self.fit)  # 输出最优值
# #         return fitness

# #         # ----------------------程序执行-----------------------


# # my_pso = PSO(pN=30, dim=1, max_iter=100)
# # my_pso.init_Population()
# # fitness = my_pso.iterator()

# # # -------------------画图--------------------
# # plt.figure(1)
# # plt.title("Figure1")
# # plt.xlabel("iterators", size=14)
# # plt.ylabel("fitness", size=14)
# # t = np.array([t for t in range(0, 100)])
# # fitness = np.array(fitness)
# # plt.plot(t, fitness, color='b', linewidth=3)
# # plt.show()



# # 随机为每个用户分配一个层级
# user_tiers = np.random.choice(num_tiers, num_clients)
# tiered_clients = {i: [] for i in range(num_tiers)}
# for user, tier in enumerate(user_tiers):
#     tiered_clients[tier].append(user)
# print(tiered_clients)

# import numpy as np
# import matplotlib.pyplot as plt

# # Define the function f(x, y) = (x + y)^2
# def f(x, y):
#     return (x + y)**2

# # Create a meshgrid for x and y
# x = np.linspace(-10, 10, 400)
# y = np.linspace(-10, 10, 400)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)

# # Plot the surface
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')

# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('f(x, y)')

# # Show the plot
# plt.show()

import cvxpy as cp
# x = cp.Variable()
# y = cp.Variable()

# constraints = [x + y ==1,
#                x - y >= 1]

# obj = cp.Minimize((x - y)**2)

# prob = cp.Problem(obj, constraints)
# prob.solve()
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x.value, y.value)


# for i, j in enumerate([1, 2, 3, 4, 5]):
#     print(i, j)

import numpy as np
import time
# ratio_B = np.random.dirichlet(np.ones(10))
# print(ratio_B)
# print(np.sum(ratio_B))

# random_numbers = np.random.uniform(low=1e-8, high=1.0, size=10)
# print(random_numbers)
# all_clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# client = 1
# # a = [1 if other_client != client else 0 for other_client in all_clients]
# # print(a)
# num_teachers = 3  # 假设需要选择3个教师
# b = np.random.choice([c for c in all_clients if c != client], size=num_teachers, replace=False)
# print(b)
# print([c for c in all_clients if c != client])

# num_teachers = np.random.randint(1, 8)
# print(num_teachers)


# s_k = np.ones((2, 2), dtype=int)
# # 使s_k中的对角线元素为0
# np.fill_diagonal(s_k, 0)
# print(s_k)



# a = cp.Variable(10, nonneg=True)
# print(a.value)

# def Indentify_idelclients(epoch, lr_tierdict):
#     idle_clients = []
#     for i in range(1, epoch + 1):
#         if i in lr_tierdict and ((epoch) % i ==0):
#             idle_clients.extend(lr_tierdict[i])
#         else:
#             print(f"Warning: Key '{i}' not found in lr_tierdict. Skipping.")
#     a = np.zeros(10)
#     for client in idle_clients:
#         a[client] = 1
#     return a

# def Indentify_idelclients_2(epoch, lr_tierdict):
#     # 只遍历存在于lr_tierdict的key，且key能整除epoch
#     idle_clients = [client for i, clients in lr_tierdict.items() if epoch % i == 0 for client in clients]
#     a = np.zeros(10, dtype=int)
#     if idle_clients:
#         a[np.array(idle_clients, dtype=int)] = 1
#     return a

# start_time = time.time()
# a = Indentify_idelclients(4, {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9]})
# end_time = time.time()
# print("运行时间: {:.6f} 秒".format(end_time - start_time))
# print(a)
# start_b_time = time.time()
# b = Indentify_idelclients_2(4, {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9]})
# end_b_time = time.time()
# print("运行时间: {:.6f} 秒".format(end_b_time - start_b_time))
# print(b)



# import numpy as np

# num_tiers = 4  # 假设有3个层级
# num_clients = 10  # 假设有10个客户端
# clients_per_tier = np.random.multinomial(num_clients, [1/num_tiers]*num_tiers)
# print(clients_per_tier)
# assert sum(clients_per_tier) == num_clients, "每层用户数之和必须等于总用户数"

# clients = list(range(num_clients))
# np.random.shuffle(clients)

# tiered_clients = {}
# start = 0
# for i, n in enumerate(clients_per_tier):
#     tiered_clients[i+1] = clients[start:start + n]
#     start += n
# print(tiered_clients)

# s_k = np.ones((2, 5), dtype=int)
# print(s_k)
# np.fill_diagonal(s_k, 0)  # 将对角线元素设为
# print(s_k)
# d = [c for c in range(5) if s_k[1][c] == 1]
# print(d)

# for i, ii in enumerate(d):
#     print(i, ii)

import cvxpy as cp
import numpy as np
from multiprocessing import Pool, cpu_count

def solve_problem(seed):
    np.random.seed(seed)
    n = 10
    A = np.random.randn(n, n)
    b = np.random.randn(n)

    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(A @ x - b))
    constraints = [x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)  # 可以改用其他 solver

    return x.value

if __name__ == '__main__':
    num_workers = min(cpu_count(), 8)  # 限制最大核数
    with Pool(num_workers) as pool:
        results = pool.map(solve_problem, range(20))  # 并行执行20个问题

    for i, res in enumerate(results):
        print(f"Result {i}: {res[:3]}")
