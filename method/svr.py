'''=================================================
@Project -> File   ：car_insurance_risk -> svr
@Author ：Zhuang Yi
@Date   ：2019/11/18 21:12
=================================================='''


from sklearn.svm import SVR
import pandas as pd
from method.pca import MY_PCA


class MY_SVR(object):
    def __init__(self):
        self.sd = MY_PCA()
        self.training_x, self.testing_x = self.sd.get_pca()
        self.training_y = self.sd.training_y
        self.id = list(range(32001, 40001))

    def sk_svm_train(self):
        linear_svr = SVR(kernel='linear')  # 线性核函数初始化的SVR
        linear_svr.fit(self.training_x, self.training_y)
        linear_svr_y_predict = linear_svr.predict(self.testing_x)

        poly_svr = SVR(kernel='poly')  # 多项式核函数初始化的SVR
        poly_svr.fit(self.training_x, self.training_y)
        poly_svr_y_predict = poly_svr.predict(self.testing_x)

        rbf_svr = SVR(kernel='rbf', C=1e2, gamma=0.1)  # 径向基核函数初始化的SVR
        rbf_svr.fit(self.training_x, self.training_y)
        rbf_svr_y_predict = rbf_svr.predict(self.testing_x)

        print('R-squared value of linear SVR is', linear_svr.score(self.training_x, self.training_y))
        print(' ')
        print('R-squared value of Poly SVR is', poly_svr.score(self.training_x, self.training_y))
        print(' ')
        print('R-squared value of RBF SVR is', rbf_svr.score(self.training_x, self.training_y))

        return linear_svr_y_predict, poly_svr_y_predict, rbf_svr_y_predict

    def test_predict(self):
        l, p, r = self.sk_svm_train()
        return r

    def write_to_file(self):
        data = self.test_predict()
        dataframe = pd.DataFrame({'Id': self.id, 'Score': data})
        dataframe.to_csv("svr_result.csv", index=False, sep=',')


if __name__ == "__main__":
    svr = MY_SVR()
    svr.write_to_file()
