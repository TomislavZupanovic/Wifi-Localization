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
supported_ap = ['TP-Link_27F9','Priva','Novinari','IoT Lab','eduroam','free_wifi','Vijecnici','predavaonice',
                'fesb-mobile','Konferencija','SDZ','Radio Data','DIRECT-A8-HP OfficeJet Pro 8730','KTOS','EMLAB']

if __name__ == '__main__':
    print('Connecting to port...\n')
    serial_port = serial.Serial(port=args.port, baudrate=9600, timeout=2)
    predictions_list = []
    counter = 0
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
                    ap_measure = data_list[::2]
                    rssi_measure = data_list[1::2]
                    indexes = [i for i, x in enumerate(ap_measure) if x in supported_ap]
                    filtered_ap = [ap_measure[i] for i in indexes]
                    filtered_rssi = [rssi_measure[i] for i in indexes]
                    ap = np.asarray(filtered_ap, dtype=np.object)
                    ap_enc = ap_encoder.transform(ap)
                    rssi = np.asarray(filtered_rssi, dtype=np.float32)
                    if ap_enc.size == rssi.size:
                        input_data = np.stack((ap_enc, rssi), axis=-1)
                        predictions = model.predict(input_data)
                        counts = np.bincount(predictions)
                        vote = np.argmax(counts)
                        room_pred = room_encoder.inverse_transform(np.asarray([vote]))
                        if counter < 5:
                            predictions_list.append(room_pred.item())
                            counter += 1
                        else:
                            # print(f'Last predictions: {predictions_list}', end='\r')
                            predictions_list = []
                            counter = 0
                        # print(f'Input: {input_data}', end='\r')
                        print(f'Prediction:  {room_pred.item()}', end='\r')
                except:
                    pass
            time.sleep(1)
    else:
        print('Serial not open.')
