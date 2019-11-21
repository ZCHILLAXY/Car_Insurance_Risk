'''=================================================
@Project -> File   ：car_insurance_risk -> correlation_test
@Author ：Zhuang Yi
@Date   ：2019/11/15 15:49
=================================================='''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

from util import TRAIN_SET_PATH, LB_TRAIN_PATH

plt.style.use('fivethirtyeight')

class TEST_Corr(object):
    def __init__(self):
        self.train_set = pd.read_csv(LB_TRAIN_PATH)
        self.training_x = self.train_set.iloc[0:24000, 1:33]
        self.validation_x = self.train_set.iloc[24000:, 1:33]
        self.training_y = self.train_set[0:24000][['Score']]
        self.validation_y = self.train_set[24000:][['Score']]
        # self.cut_training_x = self.train_set.iloc[0:24000][['Col_1', 'Col_3', 'Col_6',
        #                                                   'Col_7', 'Col_8', 'Col_9',
        #                                                   'Col_10', 'Col_11', 'Col_12',
        #                                                   'Col_14', 'Col_15',
        #                                                   'Col_16', 'Col_17', 'Col_18',
        #                                                   'Col_19',
        #                                                   'Col_24', 'Col_26',
        #                                                   'Col_27', 'Col_28', 'Col_29',
        #                                                   ]]
        # self.cut_validation_x = self.train_set.iloc[24000:][['Col_1', 'Col_3', 'Col_6',
        #                                                   'Col_7', 'Col_8', 'Col_9',
        #                                                   'Col_10', 'Col_11', 'Col_12',
        #                                                   'Col_14', 'Col_15',
        #                                                   'Col_16', 'Col_17', 'Col_18',
        #                                                   'Col_19',
        #                                                   'Col_24', 'Col_26',
        #                                                   'Col_27', 'Col_28', 'Col_29',
        #                                                   ]]


    def test_correlation(self):
        sns.heatmap(self.train_set.corr(), linewidths=.5, cmap="YlGnBu")
        plt.show()

    def my_cat(self):
        categorical_features_indices = np.where(self.training_x.dtypes != np.int64)[0]
        model = CatBoostRegressor(iterations=5000, cat_features=categorical_features_indices,
                                   learning_rate=0.01, loss_function='RMSE',
                                   logging_level='Verbose')
        model.fit(self.training_x, self.training_y, eval_set=(self.validation_x, self.validation_y), plot=True)
        fea_ = model.feature_importances_
        fea_name = model.feature_names_
        plt.figure(figsize=(10, 10))
        plt.barh(fea_name, fea_, height=0.5)
        plt.show()

    def cut_cat(self):
        categorical_features_indices = np.where(self.cut_training_x.dtypes != np.int64)[0]
        model = CatBoostRegressor(iterations=1000, cat_features=categorical_features_indices,
                                   learning_rate=0.1, loss_function='RMSE',
                                   logging_level='Verbose')
        model.fit(self.cut_training_x, self.training_y, eval_set=(self.cut_validation_x, self.validation_y), plot=True)
        fea_ = model.feature_importances_
        fea_name = model.feature_names_
        plt.figure(figsize=(10, 10))
        plt.barh(fea_name, fea_, height=0.5)
        plt.show()


if __name__ == '__main__':
    tc =TEST_Corr()
    tc.my_cat()



