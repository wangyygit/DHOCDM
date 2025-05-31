from HOCD_RL import RL
import os
import numpy as np
import torch
from utils import time_split
import pandas as pd
import argparse
#参数设置
parser = argparse.ArgumentParser(description="program")
args = parser.parse_args(args=[])
# 编码器参数
parser = argparse.ArgumentParser()

# 通用参数
parser.add_argument('--batch_size', type=int, default=9)
parser.add_argument('--input_dimension', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=12)
parser.add_argument('--length', type=int, default=5)
parser.add_argument('--nodes_num', type=int, default=5)

# 解码器参数
parser.add_argument('--decoder_hidden_dim', type=int, default=12)
parser.add_argument('--decoder_activation', type=str, default='tanh')
parser.add_argument('--use_bias', action='store_true')
parser.add_argument('--bias_initial_value', action='store_true')
parser.add_argument('--use_bias_constant', action='store_true')

# Critic 参数
parser.add_argument('--hidden_dim_critic', type=int, default=12)
parser.add_argument('--init_baseline', type=float, default=-1.0)

# Reward config
parser.add_argument('--alpha', type=float, default=0.99)

# Training config (actor)
parser.add_argument('--lr1_start', type=float, default=0.001)
parser.add_argument('--lr1_decay_rate', type=float, default=0.96)
parser.add_argument('--lr1_decay_step', type=int, default=100)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--save_file', type=str, default='Result/Simnet3')

args = parser.parse_args()


#加载数据
file_path = 'sim1/multisim1'
files = os.listdir(file_path)
num_file = len(files)
data = []
for i in range(num_file):
    #读取数据

    data_path = file_path + '/' + files[i]
    data_sam = np.load(data_path)
    print(data_sam.shape)
    #data_sam=pd.read_csv(data_path, header=None)
    data_sam = np.array(data_sam)[:,0:103]
    #data_sam = np.load(data_path)
    #print(data_sam.shape)
    #data.append(np.array(time_split(data_sam, step=5)))
    #data_sam = np.load(data_path)[:, 235:300]
    print(np.array(time_split(data_sam, step=5)).shape)
    data.append(np.array(time_split(data_sam, step=5)))

data = np.vstack(data) #
print(data.shape)#batch*nodes*length(8550,5,30)
#data = torch.FloatTensor(data)
model = RL(args)
model.learn(data)

#print(met.metrics)

