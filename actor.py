from torchdiffeq import odeint_adjoint as odeint
from PCC_compute import PCC
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
from critic import Critic


class GCN(nn.Module):
    """两层GCN模型"""
    def __init__(self, input_size, hidden_size,device):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_size, hidden_size,bias=False,dtype=torch.float32)
        self.l2 = nn.Linear(hidden_size, input_size, bias=False,dtype=torch.float32)
        self.bn = nn.BatchNorm1d(input_size,dtype=torch.float32)
        self.device = device
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)


    def forward(self, t, x):
        adj_matrix = PCC(x).to(self.device)
        output = torch.matmul(adj_matrix,x)
        output = F.relu(self.l1(output))
        output = self.l2(output)
        output = self.batch_norm(output)
        return output

class OdernnEncoder(nn.Module):
    def __init__(self, batch_size, input_dimension, hidden_dim, length, nodes_num, device):
        super(OdernnEncoder, self).__init__()

        self.batch_size = batch_size  # 256
        self.input_dimension = input_dimension  # 1
        # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.hidden_stste = hidden_dim  #
        self.length = length
        self.nodes_num = nodes_num
        self.device = device
        self.odefun = GCN(self.hidden_stste, self.hidden_stste,self.device).to(self.device)
        self.gru = nn.GRUCell(self.input_dimension, self.hidden_stste,dtype=torch.float32).to(self.device)

    def forward(self, inputs):
        #  batch* nodes* series
        inputs = inputs.unsqueeze(3).to(self.device)
        n_batch, n_node, n_ts, n_feature = inputs.shape
        #  [1,batch,hidden_dim]初始时刻的状态为0
        h_t = torch.zeros([n_batch * n_node, self.hidden_stste],dtype=torch.float32,requires_grad=True).to(self.device)
        extra_info = []
        # Run ODE backwards and combine the y(t) estimates using gating
        tt = torch.FloatTensor([0, 1]).to(self.device)
        for t in range(n_ts):
            # print(t)
            insta_t = inputs[:, :, t, :]
            h_t = self.gru(insta_t.reshape(n_batch * n_node, -1), h_t)
            h_t = h_t.reshape(n_batch, n_node, -1)
            ode_sol = odeint(self.odefun, h_t, tt, rtol=1e-6, atol=1e-12, method="euler")

            if torch.mean(ode_sol[0, :, :, :] - h_t) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[0, :, :, :] - h_t))
                exit()
            # assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)
            h_t = ode_sol[-1, :, :, :].reshape(n_batch * n_node, -1)

        return h_t.reshape(n_batch, n_node, -1)

class HypergraphDecoder(nn.Module):

    def __init__(self, batch_size,input_embed,nodes_num,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant, d, h, is_train, device=None):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.nodes_num = nodes_num    # input sequence length (number of cities)
        #self.input_dimension = input_dimension
        self.input_embed = input_embed    # dimension of embedding space (actor)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant
        self.device = device
        self.is_training = is_train
        self.edges_num = self.nodes_num           
        self.h = h
        self.d = d
        if self.decoder_activation == 'tanh':    # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self._wl = nn.Parameter(torch.FloatTensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._wr = nn.Parameter(torch.FloatTensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._u = nn.Parameter(torch.FloatTensor(*(self.decoder_hidden_dim, 1)).to(self.device))
        self._l = nn.Parameter(torch.FloatTensor(1).to(self.device))
        # 固定 N 条边的 query
        self.edge_query = nn.Parameter(torch.randn(self.edges_num, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._wl)
        nn.init.xavier_uniform_(self._wr)
        nn.init.xavier_uniform_(self._u)
        nn.init.xavier_uniform_(self.edge_query)

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = torch.randn([1]).numpy()[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        nn.init.constant_(self._l, bias_initial_value)

    def forward(self, encoder_output):
        
        # 边权重
        self.edge_weight_mlp = nn.Sequential(
                nn.Linear(self.d, self.h),
                nn.ReLU(),
                nn.Linear(self.h, 1)
                )
        w_logits = self.edge_weight_mlp(edge_q).squeeze(-1)
        w = F.softplus(w_logits) + 1e-8  # 保证正
        
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        W_l = self._wl
        W_r = self._wr
        U = self._u

        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, W_l)
        dot_r = torch.einsum('ijk, kl->ijl', encoder_output, W_r)

        tiled_l = torch.Tensor.repeat(torch.unsqueeze(dot_l, dim=2), (1, 1, self.nodes_num, 1))
        tiled_r = torch.Tensor.repeat(torch.unsqueeze(dot_r, dim=1), (1, self.nodes_num, 1, 1))
        #print(tiled_l.shape)
        #print(tiled_r.shape)
        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = torch.einsum('ijkl, l->ijk', final_sum, U.view(self.decoder_hidden_dim))  # Readability

        self.logit_bias = self._l

        if self.use_bias:  # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        self.mask = 0
        self.samples = []
        self.mask_scores = []
        self.entropy = []

        for i in range(self.nodes_num):
            position = torch.ones([encoder_output.shape[0]],
                                  device=self.device) * i
            position = position.long()

            # Update mask
            self.mask = torch.zeros((encoder_output.shape[0], self.nodes_num),
                                    device=self.device).scatter_(1, position.view(encoder_output.shape[0], 1), 1)
            self.mask = self.mask.to(self.device)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad=True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy, w

class Actor(object):
    _logger = logging.getLogger(__name__)

    def __init__(self,arg,
                 device,
                 is_train=True
            ):
        #编码器参数
        self.batch_size =arg.batch_size,
        self.input_dimension = arg.input_dimension
        self.hidden_dim = arg.hidden_dim
        self.length = arg.length
        self.nodes_num=arg.nodes_num
        self.device = device
        #解码器参数
        self.decoder_hidden_dim=arg.decoder_hidden_dim
        self.decoder_activation=arg.decoder_activation
        self.use_bias=arg.use_bias
        self.bias_initial_value=arg.bias_initial_value
        self.use_bias_constant=arg.use_bias_constant
        self.is_train=is_train
        #Critic 参数
        self.hidden_dim_critic=arg.hidden_dim_critic
        self.init_baseline=arg.init_baseline
        self.h = arg.h
        self.d = arg.d

        # Reward config
        self.avg_baseline = torch.tensor([self.init_baseline],dtype=torch.float,device=self.device)
                                          # moving baseline for Reinforce
        self.alpha = arg.alpha
        # Training config (actor)
        self.global_step = torch.Tensor([0])  # global step
        self.lr1_start       = arg.lr1_start
        self.lr1_decay_rate  = arg.lr1_decay_rate
        self.lr1_decay_step  = arg.lr1_decay_step
        # Training config (critic)
        self.global_step2 = torch.Tensor([0])  # global step
        self.lr2_start = self.lr1_start  # initial learning rate
        self.lr2_decay_rate = self.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = self.lr1_decay_step  # learning rate decay step

        # encoder
        self.encoder = OdernnEncoder(batch_size=self.batch_size,
                                         input_dimension=self.input_dimension,
                                         hidden_dim=self.hidden_dim,
                                         length=self.length,
                                         nodes_num=self.nodes_num,
                                         device=self.device)

        # decoder
        self.decoder = HypergraphDecoder(
                batch_size=self.batch_size,
                input_embed=self.hidden_dim,
                nodes_num=self.nodes_num,
                decoder_hidden_dim=self.decoder_hidden_dim,
                decoder_activation=self.decoder_activation,
                use_bias=self.use_bias,
                bias_initial_value=self.bias_initial_value,
                use_bias_constant=self.use_bias_constant,
                d=self.d,
                h=self.h,
                is_train=self.is_train,
                device=self.device)

        # critic
        self.critic = Critic(batch_size=self.batch_size,
                             nodes_num=self.nodes_num,
                             input_dimension=self.input_dimension,
                             hidden_dim=self.hidden_dim,
                             init_baseline=self.init_baseline,
                             device=self.device)
        
        # Optimizer
        self.opt1 = torch.optim.Adam([
                        {'params': self.encoder.parameters()},
                        {'params': self.decoder.parameters()},
                        {'params': self.critic.parameters()}
                    ], lr=self.lr1_start, betas=(0.9, 0.99), eps=0.0000001)

        self.lr1_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.opt1, gamma=pow(self.lr1_decay_rate, 1/self.lr1_decay_step))
        
        self.criterion = nn.MSELoss()

    def build_permutation(self, inputs):
        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = inputs
        # encoder
        self.encoder_output = self.encoder(self.input_)
        # decoder
        self.samples, self.scores, self.entropy = self.decoder(self.encoder_output)#10*64*10

        # self.samples is seq_lenthg * batch size * seq_length
        # cal cross entropy loss * reward
        graphs_gen = torch.stack(self.samples).permute([1,0,2])#64*10*10
        # graphs_gen.requires_grad = True
        self.graphs_ = graphs_gen#64*10*10

        #self.graph_batch = torch.mean(graphs_gen, axis=0)#10*10
        logits_for_rewards = torch.stack(self.scores)#10*64*10
        # logits_for_rewards.requires_grad = True
        entropy_for_rewards = torch.stack(self.entropy)#10*64*10
        # entropy_for_rewards.requires_grad = True
        entropy_for_rewards = entropy_for_rewards.permute([1, 0, 2])#64*10*10
        logits_for_rewards = logits_for_rewards.permute([1, 0, 2])#64*10*10
        self.test_scores = torch.sigmoid(logits_for_rewards)
        log_probss = F.binary_cross_entropy_with_logits(input=logits_for_rewards, 
                                                        target=self.graphs_, 
                                                        reduction='none')#64*10*10
        self.log_softmax = torch.mean(log_probss, axis=[1, 2])#torch.Size([64])
        self.entropy_regularization = torch.mean(entropy_for_rewards, axis=[1,2])#torch.Size([64])

        self.build_critic()

    def build_critic(self):
        # Critic predicts reward (parametric baseline for REINFORCE)
        self.critic = Critic(batch_size=self.batch_size,
                             nodes_num=self.nodes_num,
                             input_dimension=self.hidden_dim,
                             hidden_dim=self.hidden_dim_critic,
                             init_baseline=self.init_baseline,
                             device=self.device)
        self.critic(self.encoder_output)


    def build_reward(self, reward_):

        self.reward = reward_

        self.build_optim()

    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        # Update baseline
        reward_mean, reward_var = torch.mean(self.reward), torch.std(self.reward)
        self.reward_batch = reward_mean
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean
        self.avg_baseline = self.avg_baseline.to(self.device)
        #print(self.critic.predictions.shape)64

        # Discounted reward
        self.reward_baseline = (self.reward - self.avg_baseline - self.critic.predictions).detach()  # [Batch size, 1]

        # Loss
        self.loss1 = (torch.mean(self.reward_baseline * self.log_softmax, 0) 
                      - 1*self.lr1_scheduler.get_last_lr()[0] * torch.mean(self.entropy_regularization, 0))
                      #+ 0.09*torch.mean(torch.log(torch.abs(self.test_scores) / 1e-4 + 1)))


        self.loss2 = self.criterion(self.reward.float().detach() - self.avg_baseline, self.critic.predictions)

        # Minimize step
        self.opt1.zero_grad()
        self.loss1.backward()
        self.loss2.backward()
        
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.critic.parameters()), max_norm=1., norm_type=2)

        self.opt1.step()
        self.lr1_scheduler.step()
    def build_save(self,save_file1,save_file2):
        torch.save(self.encoder.state_dict(),save_file1)
        torch.save(self.decoder.state_dict(),save_file2)
