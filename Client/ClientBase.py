import cvxpy as cp
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict
import pdb; 


class Client(object):
    """
    This class is for train the local model with the knowledge of the teacher model, and output logits to the server
    args: argument 
    Loader_train, Loader_val, Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model,Loader_train,loader_test,idx, logger, code_length, num_classes, device, h, loader_pub):
        self.args = args
        self.logger = logger
        self.trainloader = Loader_train
        self.testloader = loader_test
        self.idx = idx
        self.ce = nn.CrossEntropyLoss() 
        self.device = device 
        self.code_length = code_length
        self.kld = nn.KLDivLoss()
        self.mse = nn.MSELoss()
        self.model = copy.deepcopy(model)
        self.clock_frequency = args.clock_frequency
        self.transmit_power = args.transmit_power
        self.h = h
        self.sigma = 1
        self.loader_pub = loader_pub
        self.data_size = len(self.loader_pub.dataset)
        self.model_size = sum(p.numel() for p in copy.deepcopy(self.model).parameters() if p.requires_grad)*32
        self.bandwidth = self.args.total_bandwidth
        self.kappa =1e-23
        self.Flops_persample = 1e6
        self.GPUFlops_percycle = 128

    def compute_loss1(self, X,y):
        X = X.to(self.device)
        y = y.to(self.device)
        _,Z = self.model(X)
        if torch.isnan(Z).any() or torch.isinf(Z).any():
            print("NaN or Inf detected in Z!")
            print("Z:", Z)
            return
        loss1 = self.ce(Z,y)
        return loss1
    
    def test_accuracy(self):
        self.model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            _, p = self.model(X)
            y_pred = p.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt

    def compute_xi(self, s_k, idle_clients):
        return 0

    # TBD
    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)

    def compute_rate(self, ratio_B):
        rate = self.bandwidth * ratio_B * np.log2(1 + (self.transmit_power * np.abs(self.h)**2) / (self.sigma**2))
        print("rate: ", rate)
        return rate

    def compute_latency(self, ratio_B, ratio_F):
        rate = self.compute_rate(ratio_B)
        transmit_latency = self.model_size/rate
        distillation_latency = (self.data_size * self.Flops_persample) / (self.GPUFlops_percycle * self.clock_frequency[self.idx] * ratio_F)
        training_latency = transmit_latency + distillation_latency
        print("Client {}:".format(self.idx))
        print("transmit_latency: {}, distillation_latency: {}".format(transmit_latency, distillation_latency))
        return training_latency
    
    def compute_energy(self, ratio_B, ratio_F):
        rate = self.compute_rate(ratio_B)
        transmit_energy = self.transmit_power * (self.model_size/rate)
        distillation_energy = self.kappa * self.data_size * self.Flops_persample * (self.GPUFlops_percycle * self.clock_frequency[self.idx] * ratio_F)**2
        training_energy  = transmit_energy + distillation_energy
        print("Client {}:".format(self.idx))
        print("transmit_energy: {}, distillation_energy: {}".format(transmit_energy, distillation_energy))
        return training_energy

    def Compute_latency_and_energy(self, ratio_B, ratio_F):
        training_latency = self.compute_latency(ratio_B, ratio_F)
        training_energy  = self.compute_energy(ratio_B, ratio_F) 
        return training_latency, training_energy


    def compute_latency_cvx(self, ratio_B, ratio_F):
        rate = self.bandwidth * ratio_B * cp.log1p((self.transmit_power * np.abs(self.h)**2) / (self.sigma**2)) / cp.log(2)
        transmit_latency = self.model_size * cp.inv_pos(rate)
        distillation_latency = (self.data_size * self.Flops_persample) / (self.GPUFlops_percycle * self.clock_frequency[self.idx] * ratio_F)
        training_latency = transmit_latency + distillation_latency
        return training_latency
    
    def compute_energy_cvx(self, ratio_B, ratio_F):
        latency = self.compute_latency_cvx(ratio_B, ratio_F)
        transmit_energy = self.transmit_power * latency
        distillation_energy = self.kappa * self.data_size * self.Flops_persample * cp.power(self.GPUFlops_percycle * self.clock_frequency[self.idx] * ratio_F, 2)
        training_energy  = transmit_energy + distillation_energy
        return training_energy