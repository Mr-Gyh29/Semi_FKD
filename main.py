import torch
import numpy as np
import os,sys,os.path
from tensorboardX import SummaryWriter
import pickle
from torch import nn
import hashlib
import argparse

from models import CNNFemnist,ResNet18,ShuffLeNet
from sampling import LocalDataset, LocalDataloaders, partition_data
from option import args_parser
from Server.ServerFKD import ServerFKD
import torchvision
from collections import Counter
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset, SubsetRandomSampler



print(torch.__version__)
torch.cuda.is_available()
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type)

args = args_parser()
print(args)
args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only':
        continue
    args_hash += str(k)+str(v)
    
args_hash = hashlib.sha256(args_hash.encode()).hexdigest()





train_dataset, testset, dict_users, dict_users_test = partition_data(
    n_users=args.num_clients,
    alpha=args.beta,
    rand_seed=args.seed,
    dataset=str(args.dataset),
    sample_method=args.sample_method
)

# 为每个用户选择与其本地数据分布相近的公共数据子集

def get_user_public_subset(train_dataset, testset, dict_user_indices, num_classes, subset_size=100):
    user_public_subsets = []
    test_labels = np.array([testset[i][1] for i in range(len(testset))])
    for user_idx, user_indices in dict_user_indices.items():
        # 统计该用户本地数据的类别分布
        user_labels = np.array([train_dataset[i][1] for i in user_indices])
        label_count = Counter(user_labels)
        # 按照用户本地分布，从testset中采样
        user_subset_indices = []
        for cls in range(num_classes):
            cls_count = label_count.get(cls, 0)
            # 从testset中找到该类别的所有索引
            cls_indices = np.where(test_labels == cls)[0]
            # 采样数量可以按比例或固定数量
            sample_num = min(len(cls_indices), max(1, int(subset_size * cls_count / max(1, len(user_labels)))))
            sampled = np.random.choice(cls_indices, sample_num, replace=False)
            user_subset_indices.extend(sampled)
        user_public_subsets.append(Subset(testset, user_subset_indices))
    return user_public_subsets

# user_public_subsets = get_user_public_subset(
#     train_dataset, testset, dict_users, args.num_classes, subset_size=100
# )
# user_public_subsets[i] 就是第i个用户的个性化公共数据集


Loaders_train = LocalDataloaders(train_dataset,dict_users,args.batch_size,ShuffleorNot = True,frac=args.part)
Loaders_test = LocalDataloaders(testset,dict_users_test,args.batch_size,ShuffleorNot = True,frac=2*args.part)
for idx, loader in enumerate(Loaders_test):
    is_empty = True
    for batch in loader:
        is_empty = False
        break
    print(f'Loaders_test[{idx}] is empty: {is_empty}')
    # 提取每个Loaders_test中一半的数据
subset_datasets = []
for loader in Loaders_test:
    dataset = loader.dataset
    indices = loader.sampler.indices if hasattr(loader.sampler, 'indices') else list(range(len(dataset)))
    half_len = int(len(indices)*0.1)
    selected_indices = np.random.choice(indices, half_len, replace=False)
    subset_datasets.append(Subset(dataset, selected_indices))

# 汇总所有子集
global_test_subset = ConcatDataset(subset_datasets)
global_loader_test = torch.utils.data.DataLoader(global_test_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
for idx, loader in enumerate(Loaders_test):
    print(f'Length of Loaders_test[{idx}]: {len(loader.dataset)}')

print(f'Length of global_loader_test: {len(global_loader_test.dataset)}')
# global_loader_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=True, num_workers=2)
images, labels = next(iter(Loaders_train[0]))
print('len(Loaders_train)', len(Loaders_train[0]))


for idx in range(args.num_clients):
    counts = [0]*args.num_classes
    for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
        batch = len(y)
        y = np.array(y)
        for i in range(batch):
            counts[int(y[i])] += 1
    print('Client {} data distribution:'.format(idx))
    print(counts)


logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+ args.dataset + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
print('Checkpoint dir:', checkpoint_dir)
img_grid = torchvision.utils.make_grid(images)
logger.add_image('images', img_grid)



print(args.model)
if args.model == 'CNN':
    # for EMNIST 62 classes
    global_model = CNNFemnist(args, code_length=args.code_len, num_classes = args.num_classes)
    
if args.model == 'resnet18':
    global_model = ResNet18(args, code_length=args.code_len, num_classes = args.num_classes)

if args.model == 'shufflenet':  
    global_model = ShuffLeNet(args, code_length=args.code_len, num_classes = args.num_classes)

   
print('# model parameters:', sum(param.numel() for param in global_model.parameters()))
# global_model = nn.DataParallel(global_model)
# logger.add_graph(global_model, torch.randn(1, 3, 224, 224).to(device))
global_model.to(device)



# if args.alg == 'FedAvg':
#     server = ServerFedAvg(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
# if args.alg == 'FedProx':
#     server = ServerFedProx(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
# if args.alg == 'FedMD':
#     server = ServerFedMD(args,global_model,Loaders_train,Loaders_test,global_loader_test,testset,logger,device)
# if args.alg == 'FedProto':    
#     server = ServerFedProto(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedKD':    
    server = ServerFKD(args,global_model,Loaders_train,Loaders_test,global_loader_test,testset,logger, device)
logger.close()

server.Create_Clints()
server.train()

save_path = checkpoint_dir + args_hash + '.pth'
if args.save_model == True:
    server.Save_CheckPoint(save_path)
    print('Model is saved on: ')
    print(save_path)






