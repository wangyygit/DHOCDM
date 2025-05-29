import numpy as np
from sklearn.preprocessing import StandardScaler

class DataGenerator(object):

    def __init__(self, X, normalize_flag=False):

        self.inputdata = X
        self.datasize, self.d, self.T = self.inputdata.shape

        if normalize_flag:
            self.inputdata = StandardScaler().fit_transform(self.inputdata)

    # Generate random batch for training procedure
    def train_batch(self, batch_size):
        seq = np.random.randint(0, self.datasize, size=(batch_size))
        input_ = self.inputdata[seq]
        return input_


