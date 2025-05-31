
import os
import logging
from tqdm import tqdm
import numpy as np
import platform
import torch
from dataset_loader import DataGenerator
from actor import Actor
from Reward import get_Reward
from utils import set_seed
from utils import convert_graph_int_to_adj_mat,evaluate_result

logger = logging.getLogger(__name__)

# 设置logger可输出日志级别范围
logger.setLevel(logging.INFO)

# 添加控制台handler，用于输出日志到控制台
console_handler = logging.StreamHandler()
# 添加日志文件handler，用于输出日志到文件中
file_handler = logging.FileHandler(filename='log.log', encoding='UTF-8')

# 将handler添加到日志器中
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 设置格式并赋予handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

class RL(object):
    """
    RL Algorithm.
    A RL-based algorithm that can work with flexible score functions (including non-smooth ones).

    """
    #@check_args_value(RL_VALID_PARAMS)
    def __init__(self, args,verbose=False,device_type='gpu',device_ids=0):
        super().__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.batch_size
        self.device_type = device_type
        self.device_ids = device_ids
        self.verbose=verbose
        self.seed=args.seed
        self.args=args
        self.length=args.length
        self.nb_epoch=args.epoch
        self.save_file=args.save_file
        if torch.cuda.is_available():
            logger.info('GPU is available.')
        else:
            logger.info('GPU is unavailable.')
            if self.device_type == 'cpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        #self.device = torch.device('cuda:1')

    def learn(self, X):
        X = torch.FloatTensor(X)
        self.data_size = X.shape[0]
        causal_matrix = self._rl(X)
        self.causal_matrix = causal_matrix

    def _rl(self, X):
        # Reproducibility
        set_seed(self.seed)
        logger.info('Python version is {}'.format(platform.python_version()))
        if not os.path.exists(self.save_file):
            os.makedirs(self.save_file)
        log_file = os.path.join(self.save_file, 'log_reward.txt')
        log = open(log_file, 'w')
        training_set = DataGenerator(X)

        # actor
        actor = Actor(self.args, device=self.device, is_train=True)
        callreward = get_Reward(self.batch_size,self.nodes_num, self.length)
        logger.info('Finished creating training dataset and reward class')

        # Initialize useful variables
        #rewards_avg_baseline = []
        #reward_max_per_batch = [] #每个batch最大奖励

        max_rewards = []
        max_reward = float('-inf')

        logger.info('Starting training.')

        #for i, inputs in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
        for j in tqdm(range(1, self.nb_epoch + 1)):

            if self.verbose:
                logger.info('Start training for {}-th epoch'.format(j))

            inputs = training_set.train_batch(self.batch_size)

            # actor
            actor.build_permutation(inputs) #64*10*10
            graphs_feed = actor.graphs_ #64*10*10

            reward_feed = callreward.cal_rewards(graphs_feed.cpu().detach().numpy(),inputs)  # 64*3

            actor.build_reward(reward_ = -torch.from_numpy(reward_feed).to(self.device))


            # max reward, max reward per batch
            #max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
            max_reward_batch = float('inf')

            m=0
            for reward_ in reward_feed:
                if reward_ < max_reward_batch:
                    max_reward_batch = reward_
                    index=m
                m=m+1
                    
            max_reward_batch = -max_reward_batch

            if max_reward < max_reward_batch:
                max_reward = max_reward_batch


            if self.verbose:
                logger.info('Finish calculating reward for current batch of graph')

            score_test, probs, graph_batch, \
            reward_batch, reward_avg_baseline = \
                    actor.test_scores, actor.log_softmax, actor.graphs_, \
                    actor.reward_batch, actor.avg_baseline

            if self.verbose:
                logger.info('Finish updating actor and critic network using reward calculated')

            max_rewards.append(max_reward)
            print('[iter {}], reward_batch: {:.4}, max_reward: {:.4}, max_reward_batch: {:.4}'.format(j,
                            reward_batch, max_reward, max_reward_batch),file=log,flush=True)
            # logging

            if j == 1 or j % 2 == 0:
                logger.info('[iter {}], reward_batch: {:.4}, max_reward: {:.4}, max_reward_batch: {:.4}'.format(j,
                            reward_batch, max_reward, max_reward_batch))

                ls_kv = callreward.update_all_scores()
                graph_int, score_min= np.array(ls_kv[0][0]), ls_kv[0][1]
                graph_batch = convert_graph_int_to_adj_mat(graph_int)
                logger.info('col.sum {}'.format(graph_batch.sum(axis=0)))
                logger.info('col.sum_tol {}'.format(graph_batch.sum()))
                save_graph=self.save_file+'/'+str(j)+'.npy'
                np.save(save_graph,graph_batch)

        logger.info('Training COMPLETED !')

        max_rewards_file=os.path.join(self.save_file, 'max_rewards.npy')
        torch.save(max_rewards, max_rewards_file)
        log.close()

        return max_rewards
