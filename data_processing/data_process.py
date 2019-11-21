'''=================================================
@Project -> File   ：car_insurance_risk -> data_process
@Author ：Zhuang Yi
@Date   ：2019/11/15 15:42
=================================================='''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from util import TRAIN_SET_PATH, TEST_SET_PATH, MY_TRAIN_PATH, MY_TEST_PATH, LB_TRAIN_PATH, LB_TEST_PATH, OH_TRAIN_PATH, \
    OH_TEST_PATH



class Feature_EX(object):
    def __init__(self):
        self.train_set = pd.read_csv(TRAIN_SET_PATH)
        self.test_set = pd.read_csv(TEST_SET_PATH)

    def my_feature_engineer(self):
        train = self.train_set.copy()
        test = self.test_set.copy()
        features = train.columns[1:]

        numeric_features = []
        categorical_features = []

        for dtype, feature in zip(train.dtypes[1:], train.columns[1:]):
            if dtype == object:
                # print(column)
                # print(train_data[column].describe())
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
        print(categorical_features)
        np.random.seed(13)
        impact_coding_map = {}
        for f in categorical_features:
            print("Impact coding for {}".format(f))
            train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = self.impact_coding(train, f)
            impact_coding_map[f] = (impact_coding_mapping, default_coding)
            mapping, default_mean = impact_coding_map[f]
            test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]]
            if x[f] in mapping else default_mean, axis=1)
        new_train = pd.DataFrame(train[list(train.columns[:])])
        new_test = pd.DataFrame(test[list(test.columns[:])])
        return new_train, new_test

    def label_feature_engineer(self):
        train = self.train_set.copy()
        test = self.test_set.copy()
        features = train.columns[1:]

        numeric_features = []
        categorical_features = []

        for dtype, feature in zip(train.dtypes[1:], train.columns[1:]):
            if dtype == object:
                # print(column)
                # print(train_data[column].describe())
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
        print(categorical_features)
        np.random.seed(13)
        le = preprocessing.LabelEncoder()
        for f in categorical_features:
            print("Impact coding for {}".format(f))
            train["impact_encoded_{}".format(f)] = le.fit_transform(train[f])
            test["impact_encoded_{}".format(f)] = le.fit_transform(test[f])
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
        train.set_index(["Id"], inplace=True)
        test.set_index(["Id"], inplace=True)
        new_train = pd.DataFrame(train[list(train.columns[:])])
        new_test = pd.DataFrame(test[list(test.columns[:])])
        return new_train, new_test

    def onehot_feature_engineer(self):
        train = self.train_set.copy()
        test = self.test_set.copy()
        features = train.columns[1:]

        numeric_features = []
        categorical_features = []

        for dtype, feature in zip(train.dtypes[1:], train.columns[1:]):
            if dtype == object:
                # print(column)
                # print(train_data[column].describe())
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
        print(categorical_features)
        np.random.seed(13)
        for f in categorical_features:
            print("Impact coding for {}".format(f))
            train = train.join(pd.get_dummies(train[f], prefix="impact_encoded_{}".format(f)))
            test = test.join(pd.get_dummies(test[f], prefix="impact_encoded_{}".format(f)))
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
        train.set_index(["Id"], inplace=True)
        test.set_index(["Id"], inplace=True)
        new_train = pd.DataFrame(train[list(train.columns[:])])
        new_test = pd.DataFrame(test[list(test.columns[:])])
        return new_train, new_test

    def write_onehotencode(self):
        ohtrain, ohtest = self.onehot_feature_engineer()
        ohtrain.to_csv(OH_TRAIN_PATH)
        ohtest.to_csv(OH_TEST_PATH)


    def write_labelencode(self):
        lbtrain, lbtest = self.label_feature_engineer()
        lbtrain.to_csv(LB_TRAIN_PATH)
        lbtest.to_csv(LB_TEST_PATH)


    def write_mydataset(self):
        mytrain, mytest = self.my_feature_engineer()
        mytrain.to_csv(MY_TRAIN_PATH)
        mytest.to_csv(MY_TEST_PATH)


    def impact_coding(self, data, feature, target='Score'):
        '''
        In this implementation we get the values and the dictionary as two different steps.
        This is just because initially we were ignoring the dictionary as a result variable.

        In this implementation the KFolds use shuffling. If you want reproducibility the cv
        could be moved to a parameter.
        '''
        n_folds = 20
        n_inner_folds = 10
        impact_coded = pd.Series()

        oof_default_mean = data[target].mean()  # Gobal mean to use by default (you could further tune this)
        kf = KFold(n_splits=n_folds, shuffle=True)
        oof_mean_cv = pd.DataFrame()
        split = 0
        for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(lambda x: oof_mean[x[feature]] if x[feature] in oof_mean.index else oof_default_inner_mean, axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1

            impact_coded = impact_coded.append(data.iloc[oof].apply(
                lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                if x[feature] in inner_oof_mean_cv.index
                else oof_default_mean
                , axis=1))

        return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean

    def run(self):
        # self.feature_engineer()
        self.write_mydataset()





if __name__ == '__main__':
    fe = Feature_EX()
    fe.run()

