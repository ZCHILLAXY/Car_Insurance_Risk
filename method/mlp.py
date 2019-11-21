'''=================================================
@Project -> File   ：car_insurance_risk -> mlp
@Author ：Zhuang Yi
@Date   ：2019/11/18 19:59
=================================================='''


import keras

from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout

from data_processing.split_dataset import Split_Train_Test
from util import MLP_MODEL_PATH, MEAN_MODEL_PATH
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

class MY_MLP(object):
    def __init__(self):
        self.sd = Split_Train_Test()
        self.training, self.testing = self.sd.split()
        self.ohtraining, self.ohtesting = self.sd.oh_split()
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.training = self.min_max_scaler.fit_transform(self.training)
        self.testing = self.min_max_scaler.fit_transform(self.testing)
        np.random.shuffle(self.training)
        self.training_x = self.training[:, :-1]
        self.testing_x = self.testing
        self.training_x_1 = self.ohtraining[:, 1:17]
        self.testing_x_1 = self.ohtesting[:, 1:17]
        self.training_x_2 = self.ohtraining[:, 17:-1]
        self.testing_x_2 = self.ohtesting[:, 17:]

        self.training_y = self.ohtraining[:, -1]
        self.id = list(range(32001, 40001))


    def mlp_train(self):
        input_1 = Input(shape=(self.training_x_1.shape[1],), name='f_in')
        x_1 = Dense(output_dim=20)(input_1)
        x_2 = Dense(output_dim=10)(x_1)
        x_out = Dense(output_dim=1, activation='softplus')(x_2)
        x_out = Dropout(0.1)(x_out)


        input_2 = Input(shape=(self.training_x_2.shape[1],), name='r_in')
        y_1 = Dense(output_dim=100)(input_2)
        y_2 = Dense(output_dim=60)(y_1)
        y_3 = Dense(output_dim=5)(y_2)
        y_out = Dense(output_dim=1, activation='softplus')(y_3)
        y_out = Dropout(0.1)(y_out)

        concatenated = keras.layers.concatenate([x_out, y_out])
        out = Dense(output_dim=1, activation='selu')(concatenated)

        merged_model = Model(inputs=[input_1, input_2], outputs=[out])

        merged_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        tensorboard = TensorBoard(log_dir='mlp_log')
        checkpoint = ModelCheckpoint(filepath=MLP_MODEL_PATH, monitor='val_loss', mode='auto')
        callback_lists = [tensorboard, checkpoint]

        history = merged_model.fit({'f_in': self.training_x_1, 'r_in': self.training_x_2}, self.training_y, verbose=2, epochs=1000, batch_size=800,
                            validation_split=0.25,
                            callbacks=callback_lists)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

        print('Train Successfully')


    def mlp_train_mean(self):
        model = Sequential()
        model.add(Dense(10,input_shape=(20,)))
        model.add(Dense(5))
        model.add(Dense(1, activation='selu'))
        model.add(Dropout(0.1))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        tensorboard = TensorBoard(log_dir='mean_log')
        checkpoint = ModelCheckpoint(filepath=MEAN_MODEL_PATH, monitor='val_loss', mode='auto')
        callback_lists = [tensorboard, checkpoint]

        history = model.fit(self.training_x, self.training_y, verbose=2,
                                   epochs=100, batch_size=800,
                                   validation_split=0.2,
                                   callbacks=callback_lists)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

    def test_predict_mean(self):
        model = load_model(MEAN_MODEL_PATH)
        y_pred = model.predict(self.testing_x)
        print(self.training_x.shape)
        print(y_pred.shape)
        after_inverse = self.min_max_scaler.inverse_transform(np.concatenate((self.testing_x[:, 1:], y_pred.reshape(-1, 1)), axis=1))
        y_pred = after_inverse[:, -1]
        for i in range(len(y_pred)):
            y_pred[i] = np.round(y_pred[i])
        print(y_pred)
        return y_pred

    def write_to_file_pca(self):
        data = self.test_predict_mean()
        dataframe = pd.DataFrame({'Id': self.id, 'Score': data})
        dataframe.to_csv("mean_result.csv", index=False, sep=',')


    def test_predict(self):
        model = load_model(MLP_MODEL_PATH)
        y_pred = model.predict({'f_in': self.testing_x_1, 'r_in': self.testing_x_2})
        for i in range(len(y_pred)):
            y_pred[i] = np.round(y_pred[i])
        print(y_pred)
        return y_pred

    def write_to_file(self):
        data = self.test_predict()
        dataframe = pd.DataFrame({'Id': self.id, 'Score': data[:, 0]})
        dataframe.to_csv("result.csv", index=False, sep=',')


if __name__ == '__main__':
    mlp = MY_MLP()
    mlp.mlp_train()
    mlp.write_to_file()