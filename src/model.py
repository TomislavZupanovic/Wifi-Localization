from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd


class Model(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.learning_rate = None
        self.optimizer = None
        self.epochs = None
        self.batch_size = None
        self.losses = None

    def build(self, learning_rate, optimizer, x_train):
        """ Builds the model with Functional API and compiles optimizer and loss function """
        normalizer = Normalization()
        normalizer.adapt(x_train)
        # Model architecture
        inputs = keras.Input(shape=self.input_shape)
        norm_layer = normalizer(inputs)
        layer1 = Dense(256, activation='elu', kernel_initializer='he_normal')(norm_layer)
        layer2 = Dense(128, activation='elu', kernel_initializer='he_normal')(layer1)
        layer3 = Dense(64, activation='elu', kernel_initializer='he_normal')(layer2)
        dropout = Dropout(0.3)(layer3)
        output = Dense(9, activation='softmax')(dropout)
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
        if self.model:
            self.epochs = epochs
            self.batch_size = batch_size
            self.losses = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                                         validation_data=(x_valid, y_valid), shuffle=False)
        else:
            raise AttributeError('First build model then train it.')

    def evaluate(self, x, y, overall=True):
        """ Evaluates model on given X features and Y true values """
        if self.model:
            prediction_arr = self.model.predict(x)
            prediction_class = np.argmax(prediction_arr, axis=1)
            accuracy = accuracy_score(y, prediction_class)
            precision = precision_score(y, prediction_class, average='micro')
            recall = recall_score(y, prediction_class, average='micro')
            if overall:
                print('Accuracy: {:.3f}'.format(accuracy))
                print('Precision: {:.3f}'.format(precision))
                print('Recall: {:.3f}'.format(recall))
            else:
                return accuracy, precision, recall
        else:
            raise AttributeError('You should first train model then evaluate it.')

    def training_curves(self):
        """ PLots training curves with training and validation loss and accuracy """
        if self.losses is None:
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
    def cm_dataframe(pred_arr, y, encoder):
        """ Calculates Confusion Matrix and makes DataFrame for plotting it """
        encoded_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        labels = encoder.inverse_transform(encoded_labels)
        prediction_class = np.argmax(pred_arr, axis=1)
        cm = confusion_matrix(y, prediction_class)
        df_cm = pd.DataFrame(cm, index=[i for i in labels],
                             columns=[i for i in labels])
        return df_cm

    def plot_confusion_matrix(self, x_train, y_train, x_valid, y_valid, x_test, y_test, encoder):
        """ Plots the Confusion Matrix for train-set, valid-set and test-set """
        if self.model:
            fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(10, 4))
            for i in range(0, 3):
                if i == 0:
                    pred_arr = self.model.predict(x_train)
                    df_cm = Model.cm_dataframe(pred_arr, y_train, encoder)
                elif i == 1:
                    pred_arr = self.model.predict(x_valid)
                    df_cm = Model.cm_dataframe(pred_arr, y_valid, encoder)
                else:
                    pred_arr = self.model.predict(x_test)
                    df_cm = Model.cm_dataframe(pred_arr, y_test, encoder)
                sn.heatmap(df_cm, annot=True, cmap="Blues", ax=axs[i])
                axs[i].set_xlabel('Predicted labels')
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


