'''=================================================
@Project -> File   ：car_insurance_risk -> optimize_parameters
@Author ：Zhuang Yi
@Date   ：2019/11/18 22:01
=================================================='''

import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from data_processing.split_dataset import Split_Train_Test


class Optimize_Paramaters(object):
    def __init__(self):
        self.sd = Split_Train_Test()
        self.training, self.testing = self.sd.split()
        self.training_x = self.training[:, :-1]
        self.testing_x = self.testing
        self.training_y = self.training[:, -1]



    # Function to create model, required for KerasClassifier
    def create_model_bat(self):
        # create model
        model = Sequential()
        model.add(Dense(30, activation='selu', input_shape=(20,)))
        model.add(Dense(10, activation='selu'))
        model.add(Dense(1, activation='selu'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

    def create_model_op(self, optimizer='adam'):
        model = Sequential()
        model.add(Dense(40, activation='selu', input_shape=(32,)))
        model.add(Dense(20, activation='selu'))
        model.add(Dense(1, activation='selu'))
        # Compile model
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model

    def create_model_act(self, activation="relu"):
        model = Sequential()
        model.add(Dense(40, activation=activation, input_shape=(32,)))
        model.add(Dense(20, activation=activation))
        model.add(Dense(1, activation='selu'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model



    def epochs_and_batch(self):
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)
        # split into input (X) and output (Y) variables
        X = self.training_x
        Y = self.training_y
        # create model
        model = KerasRegressor(build_fn=self.create_model_bat, verbose=0)
        # define the grid search parameters
        batch_size = [100, 600, 800, 1000]
        epochs = [100, 500, 1000]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        grid_result = grid.fit(X, Y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def optimizer(self):
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)
        # split into input (X) and output (Y) variables
        X = self.training_x
        Y = self.training_y
        # create model
        model = KerasRegressor(build_fn=self.create_model_op, epochs=100, batch_size=800, verbose=0)
        # define the grid search parameters
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(optimizer=optimizer)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        grid_result = grid.fit(X, Y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def activation(self):
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)
        # load dataset
        # split into input (X) and output (Y) variables
        X = self.training_x
        Y = self.training_y
        # create model
        model = KerasRegressor(build_fn=self.create_model_act, epochs=100, batch_size=800, verbose=0)
        # define the grid search parameters
        activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'selu']
        param_grid = dict(activation=activation)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        grid_result = grid.fit(X, Y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


    def run(self):
        # self.epochs_and_batch()
        # self.optimizer()
        self.activation()

if __name__ == '__main__':
    op =Optimize_Paramaters()
    op.run()








