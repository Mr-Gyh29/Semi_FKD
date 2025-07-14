import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()
    num_clients = 10
    #Data specifc paremeters
    parser.add_argument('--dataset', default='CIFAR10',
                        help='CIFAR10, CIFAR100, SVHN, EMNIST') 
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5,
                        help='frequency of logging')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--num_clients',  type=int, default=num_clients,
                        help='number of local models')   
    parser.add_argument('--num_classes', type=int,default=10,
                        help='number of classes')
    parser.add_argument('--CVX_iterations', type=int, default=20,
                        help='number of iterations for CVX solver')
    parser.add_argument('--lambda2', type=float, default=1e5, help='hyper-parameter for taylor expansion')


    parser.add_argument('--energy_threshold', type=float, default=1e6,
                        help='energy threshold for each client')
    parser.add_argument('--method_Teacher', default='Greedy',
                        help='Greedy, Random, Cosine, IL')
    parser.add_argument('--method_Bandwidth', default='CVX',
                        help='Equal, Random, Proportional, PSO, CVX')
    parser.add_argument('--method_Frequency', default='CVX',
                        help='Equal, Random, Maximum, PSO, CVX')
    parser.add_argument('--method_tier', default='Random',
                        help='Random')
    parser.add_argument('--num_tiers', type=int, default=1,
                        help='number of tiers for the clients')
    
    parser.add_argument('--duration_round', type=float, default=100, help='duration of one round')
    parser.add_argument('--sampling_rate', type=float,default=1,
                        help='frac of local models to update')
    parser.add_argument('--local_ep',type=int, default=5,
                        help='iterations of local updating')
    parser.add_argument('--beta', type=float,default=0.5,
                        help='beta for non-iid distribution')
    parser.add_argument('--seed', type=int,default=0,
                        help='random seed for generating datasets')
    parser.add_argument('--code_len', type=int,default=32,
                        help='length of code')
    parser.add_argument('--alg', default='FedKD',
                        help='FedAvg, FedProx, Moon, FedMD, Fedproto, FedDFKD')
    
    
    parser.add_argument('--lam', type=float, default=0.05,
                        help='hyper-parameter for loss2')
    
    parser.add_argument('--gamma', type=float, default=0.05,
                        help='hyper-parameter for loss3')
    
    parser.add_argument('--std', type=float, default=2,
                        help='std of gaussian noise ')
    
    parser.add_argument('--part', type=float,default=0.1,
                        help='percentage of each local data')
    
    parser.add_argument('--total_bandwidth', type=float, default=50*10**6, help='total bandwidth of the edge server')
    parser.add_argument('--clock_frequency', type=float, default=np.random.uniform(1, 10, num_clients)*1e6, help='the clock_frequency of the local device') # maximum clock frequency of the local device
    parser.add_argument('--transmit_power', type=int, default=0.5, help='the transmit power of the local device')
    
    parser.add_argument('--temp', type=float,default=1,
                        help='temperture for soft prediction')
    
    parser.add_argument('--model', default= 'shufflenet',
                        help='CNN resnet18 shufflenet')
    
    parser.add_argument('--save_model', action="store_true", default= True,
                        help='saved model parameters')
    parser.add_argument('--upload_model', action="store_true", default= True,
                        help='upload parameters')
    parser.add_argument('--eval_only', action="store_true", default=False,help='evaluate the model')

    
    args = parser.parse_args()
    return args