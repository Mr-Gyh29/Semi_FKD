
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

class Client(object):
    """
    This class is for train the local model with the knowledge of the teacher model, and output logits to the server
    args: argument 
    Loader_train, Loader_val, Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model,Loader_train,loader_test,idx, logger, code_length, num_classes, device, h):
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
        self.clock_frequency = args.clock_frequency[self.idx]
        self.transmit_power = args.transmit_power[self.idx]
        self.h = h
        self.sigma = args.sigma[self.idx]
        self.data_size = len(self.trainloader.dataset)
        self.model_size = None
        self.ratio_B = 0
        self.bandwidth = self.args.total_bandwidth * self.ratio_B

    
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

    # TBD
    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)

    def Compute_latency_and_energy(self):
        rate = self.bandwidth * np.log2(1 + (self.transmit_power * self.h) / (self.sigma**2))
        transmit_latency = self.model_size/rate
        transmit_energy = self.transmit_power * transmit_latency
        distillation_latency = (self.data_size * self.args.Flops_persample) / (self.args.GPUFlops_percycle * self.clock_frequency)
        distillation_energy = self.args.kappa * self.data_size * self.args.Flops_persample * (self.args.GPUFlops_percycle * self.clock_frequency)**2
        training_latency = transmit_latency + distillation_latency
        training_energy  = transmit_energy + distillation_energy
        return training_latency, training_energy

