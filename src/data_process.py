import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self):
        self.dataset = None
        self.model_data = None
        self.room_encoder = None
        self.ap_encoder = None

    @staticmethod
    def drop_least_ap(csv):
        """ Drops Access Points which have less than 50 data points """
        number = csv.ap_id.value_counts()
        ap_to_drop = number[number < 50].index
        idx = csv[csv.ap_id.isin(list(ap_to_drop))].index
        parsed_csv = csv.drop(idx).reset_index(drop=True)
        return parsed_csv

    @staticmethod
    def smooth_signal(csv_list):
        """ Smoothing signal with Exponential Weighted Moving Average """
        for csv in csv_list:
            csv.rssi = csv.rssi.ewm(halflife=100, min_periods=0, adjust=True).mean()

    def load_and_parse(self, list_of_files):
        """ Loads, parses and joins all data into data set """
        print('\nLoading data...')
        rooms = [pd.read_csv(f'src/Data/{room_data}') for room_data in list_of_files]
        print('Parsing data...')
        parsed_rooms = [Data.drop_least_ap(room) for room in rooms]
        Data.smooth_signal(parsed_rooms)
        self.dataset = pd.concat(parsed_rooms).reset_index(drop=True)

    def parse_for_model(self):
        """ Parsing data set in format suitable for Model"""
        print('Pre-processing data for model...')
        if self.dataset:
            model_data = self.dataset.copy()
        else:
            raise ValueError('First make dataset then pre-process it for Model')
        self.room_encoder, self.ap_encoder = LabelEncoder(), LabelEncoder()
        model_data.room_id = self.room_encoder.fit_transform(model_data.room_id)
        model_data.ap_id = self.ap_encoder.fit_transform(model_data.ap_id)
        features = model_data.iloc[:, :-1].values
        target = model_data.iloc[:, -1].values
        """ Splitting data into train, validation and test """
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
        self.model_data = (x_train, x_valid, x_test, y_train, y_valid, y_test)

