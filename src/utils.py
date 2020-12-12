def print_shapes(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('\nTrain set shape: Input {} Target {}'.format(x_train.shape, y_train.shape))
    print('Valid set shape: Input {} Target {}'.format(x_valid.shape, y_valid.shape))
    print('Test set shape: Input {} Target {}\n'.format(x_test.shape, y_test.shape))

