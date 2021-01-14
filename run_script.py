import numpy as np
import pandas as pd
import serial
import pickle
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str)
args = parser.parse_args()

print('Loading model...\n')
model = pickle.load(open('saved_model/xgb.pkl', 'rb'))
room_encoder = pickle.load(open('saved_model/room_encoder.pkl', 'rb'))
ap_encoder = pickle.load(open('saved_model/ap_encoder.pkl', 'rb'))
test_mapper = {'1e75f82b': 0, 'Zec-EXT': 1, 'Zec_EXT': 2, 'Zec': 3}

if __name__ == '__main__':
    print('Connecting to port...\n')
    serial_port = serial.Serial(port=args.port, baudrate=9600, timeout=2)
    if serial_port.is_open:
        print('Serial port open.\n')
        print('=' * 21, '\n')
        while True:
            size = serial_port.inWaiting()
            if size:
                data = serial_port.readline(size)
                data = data.decode("utf-8")
                data_list = data.split(",")
                del data_list[-1]
                try:
                    # ap = np.asarray(data_list[::2], dtype=np.object)
                    # ap_enc = room_encoder.transform(ap)
                    ap = pd.Series(data_list[::2])
                    ap_enc = ap.map(test_mapper).astype(np.int32)
                    rssi = np.asarray(data_list[1::2], dtype=np.float32)
                    if ap.size == rssi.size:
                        input_data = np.stack((ap_enc.values, rssi), axis=-1)
                        predictions = model.predict(input_data)
                        counts = np.bincount(predictions)
                        vote = np.argmax(counts)
                        room_pred = room_encoder.inverse_transform(np.asarray([vote]))
                        # print(input_data)
                        print(f'Prediction:  {room_pred.item()}', end='\r')
                except:
                    pass
            time.sleep(1)
    else:
        print('Serial not open.')
