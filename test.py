import numpy as np
import torch
import torch.nn.functional as F
# a = {'a':[1, 2, 3]}
# b = [ ]
# b.extend(a['a'])
# print(b)

# tier_client = [0, 1, 0, 1, 2, 2]
# tier_dict = {}
# for idx, tier in enumerate(tier_client):
#     if tier not in tier_dict:
#         tier_dict[tier] = []
#     tier_dict[tier].append(idx)
# print(tier_dict)

last_tierdict = {'1': [1, 2], '2': [3, 4], '3': [5, 6]}

# current_round = 3
# idle_clients = []
# # 先判断当前属于第几个回合，以及上个回合的分层情况，
# if current_round == 1:
#     print([i for i in range(4)])
# else:
#     for i in range(2, current_round+1):
#         idle_clients.extend(last_tierdict['{}'.format(i)])
#     print(idle_clients)

# a = [[1, 2],[2, 3]]
# print(np.mean(a, axis=0))

# soft_label = torch.zeros(2, 2)
# knowledge = torch.tensor([[1, 2], [3, 4]])
# print(soft_label)
# soft_label += knowledge
# print(soft_label/2)

# input1 = torch.randn(10, 100)
# input2 = torch.randn(10, 100)
# output = F.cosine_similarity(input1, input2)
# print(output.shape)
# print(output)

cosine_similarity_list = [1, 2, 4, 3, 6, 5]
sorted_indices = sorted(range(len(cosine_similarity_list)), key=lambda k: cosine_similarity_list[k], reverse=True)
print(sorted_indices)

def generate_channel_gain(num_users, variances):
    """
    生成多个用户与基站之间的信道增益 h_k。
    
    参数:
    num_users: 用户数量
    variances: 信道增益方差列表，长度应与 num_users 相同
    
    返回:
    h: 信道增益数组，形状为 (num_users,)
    """
    h = np.zeros(num_users, dtype=complex)
    for k in range(num_users):
        real_part = np.random.normal(0, np.sqrt(variances[k]))
        imag_part = np.random.normal(0, np.sqrt(variances[k]))
        h[k] = real_part + 1j * imag_part
    return h

# h = generate_channel_gain(4, [1, 1, 1, 1])
# print(np.abs(h)**2)
a = 1
b =a 
c =a
b+=1
print(c)
print(b)