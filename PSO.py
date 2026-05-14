import numpy as np
import random


# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e-10 # 全局最佳适应值

    # ---------------------目标函数-----------------------------
    # def function(self, X):
    #     return float(X**2-4*X+3)

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 1)
        self.pbest[i] = self.X[0]
        self.p_fit[i] = 0
        self.fit = 0
        self.gbest = self.X[0]

                # ----------------------更新粒子位置----------------------------------

    # def iterator(self):
    #     fitness = []
    #     for t in range(self.max_iter):
    #         for i in range(self.pN):  # 更新gbest\pbest
    #             temp = self.function(self.X[i])
    #             if temp < self.p_fit[i]:  # 更新个体最优
    #                 self.p_fit[i] = temp
    #                 self.pbest[i] = self.X[i]
    #                 if self.p_fit[i] < self.fit:  # 更新全局最优
    #                     self.gbest = self.X[i]
    #                     self.fit = self.p_fit[i]
    #         for i in range(self.pN):
    #             self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
    #                         self.c2 * self.r2 * (self.gbest - self.X[i])
    #             self.X[i] = self.X[i] + self.V[i]
    #             print("X[i]", self.X[i].shape)

    #         fitness.append(self.fit)
    #         print(self.X[0], end=" ")
    #         print(self.fit)  # 输出最优值
    #     return fitness