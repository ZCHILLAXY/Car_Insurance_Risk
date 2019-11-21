'''=================================================
@Project -> File   ：car_insurance_risk -> adaboost
@Author ：Zhuang Yi
@Date   ：2019/11/20 22:05
=================================================='''

import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

from data_processing.split_dataset import Split_Train_Test


class Ada(object):
    def __init__(self):
        self.sd = Split_Train_Test()
        self.training, self.testing = self.sd.split()
        self.training_x = self.training[:, :-1]
        self.testing_x = self.testing
        self.training_y = self.training[:, -1]
        standard_scaler = preprocessing.StandardScaler()
        self.training_x = standard_scaler.fit_transform(self.training_x)
        self.testing_x = standard_scaler.fit_transform(self.testing_x)
        self.rng = np.random.RandomState(1)
        self.id = list(range(32001, 40001))

    def AdaBoosting(self):
        regr_1 = DecisionTreeRegressor(max_depth=4)
        regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                   n_estimators=50, random_state=self.rng)
        regr_1.fit(self.training_x, self.training_y)
        regr_2.fit(self.training_x, self.training_y)
        y_1 = regr_1.predict(self.testing_x)
        y_2 = regr_2.predict(self.testing_x)
        for i in range(len(y_1)):
            y_1[i] = np.round(y_1[i])
        for i in range(len(y_2)):
            y_2[i] = np.round(y_2[i])
        data_1 = y_1
        data_2 = y_2
        dataframe = pd.DataFrame({'Id': self.id, 'Score_1': data_1, 'Score_2': data_2})
        dataframe.to_csv("ada_result.csv", index=False, sep=',')
if __name__ == '__main__':
    ada = Ada()
    ada.AdaBoosting()
