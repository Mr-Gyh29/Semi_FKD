
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict
from Client.ClientBase import Client
import gc
import pdb; 

class ClientFKD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, Loader_train, loader_test, Loader_global_test, loader_pub,idx, logger, code_length, num_classes, device, h):
        super().__init__(args, model, Loader_train,loader_test,Loader_global_test,idx, logger, code_length, num_classes, device, h, loader_pub)
        self.loader_pub = loader_pub
        
    def update_weights(self,global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _,p = self.model(X)
                loss = self.ce(p,y)               
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('batch_idx:', batch_idx)
                    print('len(X):', len(X))
                    print('| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.idx, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    # 弄清楚knowledges的来源，参考原代码
    def Local_train(self,knowledges, lam, temp, global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        epoch_loss1 = []
        global_soft_prediction =  torch.stack(knowledges)
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss1 = []
            for batch_idx, (X, y) in enumerate(self.trainloader):  
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!batch_idx:", batch_idx)            
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _,Z = self.model(X)
                if torch.isnan(Z).any() or torch.isinf(Z).any():
                    print("NaN or Inf detected in Z!")
                    print("Z:", Z)
                    return
                if self.args.loss_F == "CEKD":
                    # loss1 = self.ce(Z,y)
                    loss1 = F.cross_entropy(Z, y)
                else:
                    y_onehot = F.one_hot(y, num_classes=self.args.num_classes).float()
                    loss1 = F.mse_loss(Z, y_onehot)
                loss2 = torch.tensor(0.0).to(self.device)
                total_sum = sum([k.sum().item() for k in knowledges])
                # print("total_sum:", total_sum)
                if total_sum != 0:
                    for idx, (X_pub,y_pub) in enumerate(self.loader_pub):
                        # 检查
                        if idx == batch_idx:
                            X_pub = X_pub.to(self.device)
                            y_pub = y_pub.to(self.device)
                            _,Z_pub = self.model(X_pub)
                            Q_pub = soft_predict(Z_pub, temp).to(self.device)
                                    # 检查输入是否有 NaN 或 Inf
                            if torch.isnan(Q_pub).any() or torch.isinf(Q_pub).any():
                                print("NaN or Inf detected in Q_pub!")
                                return

                            if torch.isnan(global_soft_prediction[idx]).any() or torch.isinf(global_soft_prediction[idx]).any():
                                print("NaN or Inf detected in global_soft_prediction!")
                                return
                            
                            if self.args.loss_F == "CEKD":
                                loss2 -= F.kl_div(Q_pub,global_soft_prediction[idx].to(self.device), reduction='batchmean')
                            else:
                                loss2 += F.mse_loss(Q_pub, global_soft_prediction[idx].to(self.device))
                if torch.isnan(loss1) or torch.isnan(loss2) or torch.isinf(loss1) or torch.isinf(loss2):
                    pdb.set_trace()
                loss = loss1 + lam*loss2
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Client :{} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.6f} Loss2: {:.6f} '.format(
                        global_round, self.idx, iter, (batch_idx+1) * len(X),
                        len(self.trainloader.dataset),
                        100. * (batch_idx+1) / len(self.trainloader), loss1.item(),loss2.item()))
                batch_loss.append(loss.item())
                batch_loss1.append(loss1.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_loss1.append(sum(batch_loss1)/len(batch_loss1))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss1) / len(epoch_loss1)

    def generate_knowledge(self, temp):
        self.model.to(self.device)
        self.model.eval()
        num_classes = self.model.num_classes
        soft_predictions = []
        for batch_idx, (X, y) in enumerate(self.loader_pub):
            X = X.to(self.device)
            y = y
            _,Z = self.model(X) 
            Q = soft_predict(Z,temp).to(self.device).detach().cpu() # the dimension of Q is [batch_size, num_classes]
            soft_predictions.append(Q) # the dimension of soft_predictions is [batch_num, batch_size, num_classes]
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("NaN or Inf detected in Q!")
                print("Q:", Q)
                print("Z:", Z)
                return None
            del X
            del y
            del Z
            del Q
            gc.collect()
         
        return soft_predictions

    