import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
from torchvision import datasets, transforms

class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
# 载入数据集
def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True, frac = 1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        frac_num_data = int(frac*num_data)
        whole_range = range(num_data)
        frac_range = np.random.choice(whole_range, frac_num_data)
        frac_dict_users = [dict_users[i][j] for j in frac_range]
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        loaders.append(loader)
    return loaders


# 划分数据
def partition_data(n_users, alpha=0.5,rand_seed = 0, dataset = 'cifar10'):
    def partition_data(n_users, alpha=0.5, rand_seed=0, dataset='cifar10'):
        """
        Partitions the dataset into non-IID subsets for federated learning.
        Parameters:
        n_users (int): Number of users to partition the data among.
        alpha (float, optional): Parameter for the Dirichlet distribution to control the degree of non-IID. Default is 0.5.
        rand_seed (int, optional): Random seed for reproducibility. Default is 0.
        dataset (str, optional): The name of the dataset to partition. Options are 'CIFAR10', 'CIFAR100', 'EMNIST', 'SVHN'. Default is 'cifar10'.
        Returns:
        tuple: A tuple containing:
            - train_dataset (Dataset): The training dataset.
            - test_dataset (Dataset): The testing dataset.
            - net_dataidx_map (dict): A dictionary mapping user indices to their respective training data indices.
            - net_dataidx_map_test (dict): A dictionary mapping user indices to their respective testing data indices.
        """
    if dataset == 'CIFAR10':  # Check if the dataset is CIFAR10
        K = 10  # Number of classes in CIFAR10
        data_dir = '../data/cifar10/'  # Directory to store CIFAR10 data
        apply_transform = transforms.Compose(  # Define the data transformation
            [transforms.ToTensor(),  # Convert images to tensor
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize the images
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,  # Load CIFAR10 training dataset
                                       transform=apply_transform)  # Apply transformations
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,  # Load CIFAR10 test dataset
                                          transform=apply_transform)  # Apply transformations
        y_train = np.array(train_dataset.targets)  # Get training labels
        y_test = np.array(test_dataset.targets)  # Get test labels
        
    if dataset == 'CIFAR100':
        K = 100
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
    if dataset == 'EMNIST':
        K = 62
        data_dir = '../data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.EMNIST(data_dir, train=True, split = 'byclass', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split = 'byclass', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
    if dataset == 'SVHN':
        K = 10
        data_dir = '../data/SVHN/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.labels) # Get training labels
        y_test = np.array(test_dataset.labels)
        
    min_size = 0  # Initialize the minimum size of data assigned to any user
    N = len(train_dataset)  # Total number of training samples
    N_test = len(test_dataset)  # Total number of test samples
    net_dataidx_map = {}  # Dictionary to store training data indices for each user
    net_dataidx_map_test = {}  # Dictionary to store test data indices for each user
    np.random.seed(rand_seed)  # Set random seed for reproducibility

    while min_size < 10:  # Ensure each user gets at least 10 samples
        idx_batch = [[] for _ in range(n_users)]  # Initialize list to store training indices for each user
        idx_batch_test = [[] for _ in range(n_users)]  # Initialize list to store test indices for each user
        for k in range(K):  # Iterate over each class
            idx_k = np.where(y_train == k)[0]  # Get all training indices for class k
            idx_k_test = np.where(y_test == k)[0]  # Get all test indices for class k
            np.random.shuffle(idx_k)  # Shuffle the training indices
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))  # Sample proportions for each user from a Dirichlet distribution
            # Adjust proportions to ensure balance
            proportions_train = np.array([p * (len(idx_j) < N / n_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions_test = np.array([p * (len(idx_j) < N_test / n_users) for p, idx_j in zip(proportions, idx_batch_test)])
            proportions_train = proportions_train / proportions_train.sum()  # Normalize proportions for training data
            proportions_test = proportions_test / proportions_test.sum()  # Normalize proportions for test data
            proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]  # Compute cumulative sum for training data
            proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]  # Compute cumulative sum for test data
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]  # Split and assign training indices to users
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]  # Split and assign test indices to users
            min_size = min([len(idx_j) for idx_j in idx_batch])  # Update the minimum size of data assigned to any user

    for j in range(n_users):  # Iterate over each user
        np.random.shuffle(idx_batch[j])  # Shuffle the training indices for user j
        net_dataidx_map[j] = idx_batch[j]  # Assign the shuffled training indices to the user
        np.random.shuffle(idx_batch_test[j])  # Shuffle the test indices for user j
        net_dataidx_map_test[j] = idx_batch_test[j]  # Assign the shuffled test indices to the user
   
        
    return (train_dataset, test_dataset,net_dataidx_map, net_dataidx_map_test)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts
