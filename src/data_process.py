import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self):
        self.dataset = None
        self.model_data = None
        self.parsed_rooms = None
        self.room_encoder = None
        self.ap_encoder = None

    @staticmethod
    def drop_least_ap(csv):
        """ Drops Access Points which have less than 2000 data points """
        number = csv.ap_id.value_counts()
        ap_to_drop = number[number < 2000].index
        idx = csv[csv.ap_id.isin(list(ap_to_drop))].index
        parsed_csv = csv.drop(idx).reset_index(drop=True)
        return parsed_csv

    @staticmethod
    def smooth_signal(csv_list):
        """ Smoothing signal of each Access Point with Exponential Weighted Moving Average """
        for csv in csv_list:
            ap_list = list(csv.ap_id.unique())
            for ap in ap_list:
                cut = csv.loc[csv.ap_id == ap]
                cut.rssi = cut.rssi.ewm(halflife=250, min_periods=0, adjust=True).mean()
                csv.update(cut)

    def load_and_parse(self, list_of_files):
        """ Loads, parses and joins all data into data set """
        print('\nLoading data...')
        rooms = [pd.read_csv(f'src/Data/{room_data}') for room_data in list_of_files]
        print('Parsing data...')
        Data.smooth_signal(rooms)
        dataset = pd.concat(rooms).reset_index(drop=True)
        self.dataset = Data.drop_least_ap(dataset)
        self.parsed_rooms = [self.dataset.loc[self.dataset.room_id == room].reset_index(drop=True)
                             for room in list(self.dataset.room_id.unique())]

    def parse_for_model(self):
        """ Parsing data set in format suitable for Model"""
        print('Pre-processing data for model...')
        if any(self.dataset):
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


