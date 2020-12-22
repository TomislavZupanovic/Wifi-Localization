from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import os


class Model(object):
    def __init__(self):
        self.input_shape = None
        self.model = None
        self.learning_rate = None
        self.optimizer = None
        self.epochs = None
        self.batch_size = None
        self.losses = None

    def build(self, learning_rate, optimizer, input_shape, x_train):
        """ Builds the model with Functional API and compiles optimizer and loss function """
        self.input_shape = input_shape
        normalizer = Normalization()
        normalizer.adapt(x_train)
        # Model architecture
        inputs = keras.Input(shape=self.input_shape)
        norm_layer = normalizer(inputs)
        layer1 = Dense(128, activation='elu', kernel_initializer='he_normal')(norm_layer)
        layer2 = Dense(64, activation='elu', kernel_initializer='he_normal')(layer1)
        layer3 = Dense(32, activation='elu', kernel_initializer='he_normal')(layer2)
        dropout = Dropout(0.4)(layer3)
        output = Dense(7, activation='softmax')(dropout)
        self.model = keras.Model(inputs=inputs, outputs=output)

        self.learning_rate = learning_rate
        if optimizer == 'sgd':
            optim = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer == 'rmsprop':
            optim = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
        elif optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise ValueError('No such optimizer.')
        self.model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('Model built!\n')
        print(self.model.summary(), '\n')
        print('=' * 70)

    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch_size):
        """ Trains and validates the built model """
        if self.model is not None:
            self.epochs = epochs
            self.batch_size = batch_size
            self.losses = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                                         validation_data=(x_valid, y_valid), shuffle=False)
        else:
            raise AttributeError('First build model then train it.')

    def evaluate(self, x, y, overall=True):
        """ Evaluates model on given X features and Y true values """
        if self.losses is not None:
            prediction_arr = self.model.predict(x)
            prediction_class = np.argmax(prediction_arr, axis=1)
            accuracy = accuracy_score(y, prediction_class)
            f1 = f1_score(y, prediction_class, average='macro')
            if overall:
                print('Accuracy: {:.3f}'.format(accuracy))
                print('F1 score: {:.3f}'.format(f1))
            else:
                return accuracy, f1
        else:
            raise AttributeError('You should first train model then evaluate it.')

    def evaluate_every_room(self, rooms, room_encoder, ap_encoder):
        """ Evaluates model on every room separately """
        if not self.losses:
            raise AttributeError('You should first train model then evaluate it.')
        index = [room.room_id[0] for room in rooms]
        columns = ['Accuracy', 'F1 score']
        res_df = pd.DataFrame(index=index, columns=columns)
        for room in rooms:
            room_name = room.room_id[0]
            room.ap_id = ap_encoder.transform(room.ap_id)
            room.room_id = room_encoder.transform(room.room_id)
            x = room.iloc[:, :-1]
            y = room.iloc[:, -1]
            accuracy, f1 = self.evaluate(x.values, y.values, overall=False)
            res_df.loc[room_name] = [round(accuracy, 3), round(f1, 3)]
        print('\n==== EVALUATION FOR EVERY ROOM ===\n\n', res_df, '\n')

    def training_curves(self):
        """ Plots training curves with training and validation loss and accuracy """
        if not self.losses:
            raise AttributeError('You have not trained your model!')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(12, 5))
            """ Losses plot """
            axs[0].plot(self.losses.history['loss'], 'b-', label='Train')
            axs[0].plot(self.losses.history['val_loss'], 'r-', label='Validation')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Loss')
            axs[0].legend(loc='best')
            axs[0].set_title('Learning curves')
            axs[0].grid()
            """ Accuracy plot """
            axs[1].plot(self.losses.history['accuracy'], 'b-', label='Train')
            axs[1].plot(self.losses.history['val_accuracy'], 'r-', label='Validation')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            axs[1].legend(loc='best')
            axs[1].set_title('Accuracy plot')
            axs[1].grid()
            plt.show()

    @staticmethod
    def cm_dataframe(pred_arr, y, room_encoder):
        """ Calculates Confusion Matrix and makes DataFrame for plotting it """
        encoded_labels = [i for i in range(len(os.listdir('data/')))]
        labels = room_encoder.inverse_transform(encoded_labels)
        prediction_class = np.argmax(pred_arr, axis=1)
        cm = confusion_matrix(y, prediction_class)
        df_cm = pd.DataFrame(cm, index=[i for i in labels],
                             columns=[i for i in labels])
        return df_cm

    def plot_confusion_matrix(self, x_train, y_train, x_valid, y_valid, x_test, y_test, room_encoder):
        """ Plots the Confusion Matrix for train-set, valid-set and test-set """
        if self.model:
            fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(18, 5))
            for i in range(0, 3):
                if i == 0:
                    pred_arr = self.model.predict(x_train)
                    df_cm = Model.cm_dataframe(pred_arr, y_train, room_encoder)
                elif i == 1:
                    pred_arr = self.model.predict(x_valid)
                    df_cm = Model.cm_dataframe(pred_arr, y_valid, room_encoder)
                else:
                    pred_arr = self.model.predict(x_test)
                    df_cm = Model.cm_dataframe(pred_arr, y_test, room_encoder)
                sn.heatmap(df_cm, annot=True, cmap="Blues", ax=axs[i], fmt='g')
                axs[i].set_ylabel('True labels')
                if i == 0:
                    axs[i].set_title('Training')
                elif i == 1:
                    axs[i].set_title('Validation')
                else:
                    axs[i].set_title('Test')
            plt.show()
        else:
            raise AttributeError('First build and train model.')


