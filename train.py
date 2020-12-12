import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.data_process import Data
from src.model import Model
from src.utils import print_shapes
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='adam', help='Choose optimizer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
args = parser.parse_args()

data_list = os.listdir('src/data/')
data = Data()
model = Model()

if __name__ == '__main__':
    data.load_and_parse(data_list)
    data.parse_for_model()
    x_train, x_valid, x_test, y_train, y_valid, y_test = data.model_data
    model.build(learning_rate=args.learning_rate, optimizer=args.optimizer,
                input_shape=(x_train.shape[1],), x_train=x_train)
    print_shapes(x_train, y_train, x_valid, y_valid, x_test, y_test)
    start_button = input('Start training? [y,n] ')
    if start_button == 'y':
        model.train(x_train, y_train, x_valid, y_valid, args.epochs, args.batch_size)
        print('\n\n==== OVERALL EVALUATION ====\n\nTRAINING:')
        model.evaluate(x_train, y_train)
        print('\n\nVALIDATION:')
        model.evaluate(x_valid, y_valid)
        print('\n\nTEST:')
        model.evaluate(x_test, y_test)
        model.evaluate_every_room(data.parsed_rooms, data.room_encoder, data.ap_encoder)
        model.training_curves()
        plot_button = input('\nPlot Confusion Matrix? [y,n] ')
        if plot_button == 'y':
            model.plot_confusion_matrix(x_train, y_train, x_valid, y_valid, x_test, y_test, data.room_encoder)
        else:
            pass
        save_button = input('\nSave model? [y,n] ')
        if save_button == 'y':
            model.model.save('saved_models/Model')
        else:
            pass
    else:
        pass
