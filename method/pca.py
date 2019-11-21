'''=================================================
@Project -> File   ：car_insurance_risk -> pca
@Author ：Zhuang Yi
@Date   ：2019/11/19 20:07
=================================================='''

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from util import OH_TRAIN_PATH, OH_TEST_PATH


class MY_PCA(object):
    def __init__(self):
        self.training = pd.read_csv(OH_TRAIN_PATH).values
        self.testing = pd.read_csv(OH_TEST_PATH).values
        self.training_x = self.training[:, :-1]
        self.testing_x = self.testing
        self.training_y = self.training[:, -1]
        self.id = list(range(32001, 40001))

    def get_pca(self):
        Matrix = np.concatenate((self.training_x, self.testing_x), axis=0)
        minMax = MinMaxScaler()
        Matrix = minMax.fit_transform(Matrix)
        estimator = PCA(n_components=0.9)
        X_pca = estimator.fit_transform(Matrix)
        training_pca = X_pca[0:32000, :]
        testing_pca = X_pca[-8000:, :]
        return training_pca, testing_pca

if __name__ == '__main__':
    c = MY_PCA()
    c.get_pca()