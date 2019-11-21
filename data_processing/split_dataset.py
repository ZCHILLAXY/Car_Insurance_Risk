'''=================================================
@Project -> File   ：car_insurance_risk -> split_dataset
@Author ：Zhuang Yi
@Date   ：2019/11/18 20:01
=================================================='''

import pandas as pd

from util import MY_TRAIN_PATH, MY_TEST_PATH, LB_TEST_PATH, LB_TRAIN_PATH, OH_TRAIN_PATH, OH_TEST_PATH


class Split_Train_Test(object):
    def __init__(self):
        self.train_set = pd.read_csv(MY_TRAIN_PATH)
        self.test_set = pd.read_csv(MY_TEST_PATH)
        self.ohtrain_set = pd.read_csv(OH_TRAIN_PATH)
        self.ohtest_set = pd.read_csv(OH_TEST_PATH)
        self.cut_training = self.train_set.iloc[:][['Col_1', 'impact_encoded_Col_3', 'Col_6',
                                                          'Col_7', 'impact_encoded_Col_8', 'impact_encoded_Col_9',
                                                          'Col_10', 'impact_encoded_Col_11', 'impact_encoded_Col_12',
                                                          'Col_14', 'impact_encoded_Col_15',
                                                          'Col_16', 'impact_encoded_Col_17', 'Col_18',
                                                          'Col_19',
                                                          'impact_encoded_Col_24', 'Col_26',
                                                          'impact_encoded_Col_27', 'impact_encoded_Col_28', 'impact_encoded_Col_29',
                                                          'Score']]
        self.cut_testing = self.test_set.iloc[:][['Col_1', 'impact_encoded_Col_3', 'Col_6',
                                                          'Col_7', 'impact_encoded_Col_8', 'impact_encoded_Col_9',
                                                          'Col_10', 'impact_encoded_Col_11', 'impact_encoded_Col_12',
                                                          'Col_14', 'impact_encoded_Col_15',
                                                          'Col_16', 'impact_encoded_Col_17', 'Col_18',
                                                          'Col_19',
                                                          'impact_encoded_Col_24', 'Col_26',
                                                          'impact_encoded_Col_27', 'impact_encoded_Col_28', 'impact_encoded_Col_29',
                                                           ]]

    def split(self):
        training = self.cut_training.values
        testing = self.cut_testing.values
        return training, testing

    def oh_split(self):
        training = self.ohtrain_set.values
        testing = self.ohtest_set.values
        return training, testing

    def run(self):
        self.split()



if __name__ == '__main__':
    stt = Split_Train_Test()
    stt.run()