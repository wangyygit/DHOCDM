# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
from sklearn.linear_model import LinearRegression


class get_Reward(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, batch_num, nodes_num, len, verbose_flag=False):
        self.batch_num = batch_num
        self.nodes_num = nodes_num  # =d: number of vars
        self.length = len
        self.bic_penalty = np.log(self.batch_num) / self.batch_num
        self.baseint = 2**nodes_num
        self.verbose = verbose_flag
        self.d = {}
        self.d_RSS = {}
        self.ones = np.ones((batch_num, 1), dtype=np.float32)
    def cal_rewards(self, graphs, inputdata):  # 64*5*5, 64*5*6
        rewards_batches = []
        inputdata = inputdata.numpy()
        for i in range(graphs.shape[0]):
            reward_ = self.calculate_reward_single_graph(graphs[i], inputdata)
            rewards_batches.append(reward_)

        return np.array(rewards_batches)

    ####### regression



    # faster than LinearRegression() from sklearn
    def calculate_LR(self, X_train, y_train):
        regr = LinearRegression(fit_intercept=False)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        y_err = y_pred - y_train
        return y_err



    ####### score calculations

    def calculate_reward_single_graph(self, graph_batch, inputdata):
        graph_to_int = []
        graph_to_int2 = []
        for i in range(len(graph_batch)):
            graph_batch[i][i] = 0
            tt = np.int32(graph_batch[:, i])
            graph_to_int.append(self.baseint * i + int(''.join([str(ad) for ad in tt]), 2))
            graph_to_int2.append(int(''.join([str(ad) for ad in tt]), 2))

        graph_batch_to_tuple = tuple(graph_to_int2)

        if graph_batch_to_tuple in self.d:
            score = self.d[graph_batch_to_tuple]
            return score

        RSS_ls = []
        aux_matrix = 1 / (np.diag(np.sum(graph_batch, axis=1)) + 0.000001)
        aux_matrix = np.where(aux_matrix > 1, 0, aux_matrix)
        aux_matrix = np.matmul(aux_matrix, graph_batch)
        for i in range(len(graph_batch)):
            col = graph_batch[:, i]
            aux_matrix_col = aux_matrix[:, i].reshape(1, -1)
            if graph_to_int[i] in self.d_RSS:
                RSS_ls.append(self.d_RSS[graph_to_int[i]])
                continue
            if np.sum(col) < 0.1:
                y_err = inputdata[:, :, -1].reshape(-1, self.nodes_num)[:, i]
                y_err = y_err - np.mean(y_err)
            else:
            # no parents, then simply use mean
                cols_TrueFalse = col > 0.5
                X_train = inputdata[:, :, -2].reshape(-1, self.nodes_num) * aux_matrix_col
                X_train = X_train[:, cols_TrueFalse]
                y_train = inputdata[:, :, -1].reshape(-1, self.nodes_num)[:, i]
                y_err = self.calculate_LR(X_train, y_train)

            RSSi = np.sum(np.square(y_err))

            RSS_ls.append(RSSi)
            self.d_RSS[graph_to_int[i]] = RSSi

        epsilon = 1e-4
        score = np.log(np.sum(RSS_ls) / self.batch_num + 1e-8) + 0.01* np.mean(np.log(np.abs(graph_batch) / epsilon + 1))
        self.d[graph_batch_to_tuple] = (score)

        if self.verbose:
            self._logger.info('returned reward: {}'.format( score))

        return score


    def update_scores(self, score):
        ls = []
        for score_ in score:
            ls.append(score_)
        return ls

    def update_all_scores(self):
        score_s = list(self.d.items())
        ls = []
        for graph_int, score_ in score_s:
            ls.append((graph_int, score_))
        return sorted(ls, key=lambda x: x[1])
