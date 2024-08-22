import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from datetime import datetime
import pickle
import numpy as np


from models import *
import time
from threading import Thread
import pickle

import torch
import csv



from arch import arch_model
import torch
from statsmodels.tsa.arima.model import ARIMA
import os

logger = logging.getLogger()
TF_EQUIV = {"1m": 60, '3m':180,"5m": 300, "15m": 900, "30m": 900, "1h": 3600, "4h": 14400,"1d":86400,"3d":259200,"1w":604800,"1M":2592000}

class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim, first_conv_dim, first_conv_kernel, first_conv_activation, first_lstm_dim, first_dense_dim, first_dense_activation, output_dim, gru_hidden_size, attention_dim):
        super(ComplexLSTMModel, self).__init__()

        # First Conv1D layer
        self.conv1d_1 = nn.Conv1d(in_channels=input_dim, out_channels=first_conv_dim, kernel_size=first_conv_kernel, padding='same')
        self.conv_activation_1 = nn.ReLU() if first_conv_activation == 'relu' else nn.Tanh()

        # Second Conv1D layer
        self.conv1d_2 = nn.Conv1d(in_channels=first_conv_dim, out_channels=first_conv_dim, kernel_size=first_conv_kernel, padding='same')
        self.conv_activation_2 = nn.ReLU() if first_conv_activation == 'relu' else nn.Tanh()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=first_conv_dim, hidden_size=first_lstm_dim, batch_first=True, bidirectional=True)

        # Dense layers
        self.dense1 = nn.Linear(first_lstm_dim * 2, first_dense_dim)  # Adjust input size for bidirectional LSTM
        self.dense_activation1 = nn.ReLU() if first_dense_activation == 'relu' else nn.Tanh()

        self.dense2 = nn.Linear(first_dense_dim, first_dense_dim)
        self.dense_activation2 = nn.ReLU() if first_dense_activation == 'relu' else nn.Tanh()

        self.dense3 = nn.Linear(first_dense_dim, output_dim)

    def forward(self, x):
        # First Conv1D and activation
        x = self.conv1d_1(x.transpose(1, 2))  # Adjust dimensions for Conv1d
        x = self.conv_activation_1(x)

        # Second Conv1D and activation
        x = self.conv1d_2(x)
        x = self.conv_activation_2(x)

        # LSTM layer
        x, (hn, cn) = self.lstm(x.transpose(1, 2))  # Adjust dimensions for LSTM
        lstm_features = x[:, -1, :]  # Take the output of the last time step

        # Dense layers with activations
        x = self.dense1(lstm_features)
        x = self.dense_activation1(x)

        x = self.dense2(x)
        x = self.dense_activation2(x)

        x = self.dense3(x)
        return x, lstm_features




class LSTM_BTC():
    def __init__(self, client, contract: Contract, timeframe: str,plus_1  = False):
        print("initializing LSTM")
        if plus_1:
            self.result_csv_path = f"predictionsResults/hybrid_LSTM_{timeframe}_plus_1.csv"
        else:
            self.result_csv_path = f"predictionsResults/hybrid_LSTM_{timeframe}_sub_1.csv"
        
        self.client = client
        self.contract = contract
        self.tf = timeframe
        self.tf_equiv = TF_EQUIV[timeframe] * 1000
        self.trades = []

        headers = ['timestamp','prediction','entry','suggested_exit']
        if not os.path.exists(self.result_csv_path):
            # Open the file in write mode
            with open(self.result_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write the headers
                writer.writerow(headers)            

        # add if else here for different weights for different time frames
        if plus_1:
            model_state_dict = torch.load(f"saved_models\Complete_OHLCAG_Hybrid_Model_{self.tf}_plus_1.pth")    

            with open(f'saved_models\linear_regression_model_{self.tf}_plus_1.pkl', 'rb') as f:
                self.regressor = pickle.load(f)

        else:
            model_state_dict = torch.load(f"saved_models\Complete_OHLCAG_Hybrid_Model_{self.tf}_sub_1.pth")    
            
            with open(f'saved_models\linear_regression_model_{self.tf}_sub_1.pkl', 'rb') as f:
                self.regressor = pickle.load(f)
        
        
        self.exchange = 'Binance'

        # Load the saved model

        if self.contract.candlesRetrieved[self.tf]:
            self.last_candle = self.contract.candles[self.tf][-1]

        model_params = {
            'input_dim': 6,  # Number of features
            'first_conv_dim': 64,
            'first_conv_kernel': 3,
            'first_conv_activation': 'relu',
            'first_lstm_dim': 32,
            'first_dense_dim': 64,
            'first_dense_activation': 'relu',
            'output_dim': 1,  # Close price
            'gru_hidden_size': 32,  # Optimal GRU hidden size
            'attention_dim': 32  # Optimal attention dimension
        }
        self.model = ComplexLSTMModel(**model_params)

        # Load the model's state dictionary
        self.model.load_state_dict(model_state_dict)

        # Set the model to evaluation mode
        self.model.eval()

        

        self.contract = contract


        self.parseMutex = False
        self.predictMutex = False

        self._predict()
    




    def parse_trades(self, data) -> str:
        if self.tf != data['i']:
            return "Different TF"
        if self.parseMutex:
            return
        self.parseMutex =True

        last_candle = self.last_candle


        timestamp_kline = float(data['t'])

        if timestamp_kline < last_candle.timestamp + self.tf_equiv:

            last_candle.close = float(data['c'])
            last_candle.high = float(data['h'])
            last_candle.low = float(data['l'])

            self.parseMutex = False
            return "same_candle"

        # Missing Candle(s)

        elif timestamp_kline >= last_candle.timestamp + 2 * self.tf_equiv:

            missing_candles = int((timestamp_kline - last_candle.timestamp) / self.tf_equiv) - 1

            logger.info("%s missing %s candles for %s %s (%s %s)", self.exchange, missing_candles, self.contract.symbol,
                        self.tf, timestamp_kline, last_candle.timestamp)

            candles  = self.client.get_historical_candles(self.contract, self.tf,limit=1000)



            self.contract.candles[self.tf] = candles
            self.last_candle =  self.contract.candles[self.tf][-1]
            self.parseMutex = False
            return "new_candle"

        # New Candle
        elif timestamp_kline >= last_candle.timestamp + self.tf_equiv:
            
            new_ts = timestamp_kline
            candle_info = {'ts': new_ts, 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['q']}
            new_candle = Candle(candle_info, self.tf, "parse_trade")

            if timestamp_kline >= self.contract.candles[self.tf][-1].timestamp + self.tf_equiv:
                logger.info("%s New candle for %s %s Latest Candle Timestamp %s ,RSI %s", self.exchange, self.contract.symbol, self.tf,new_candle.timestamp,self.last_candle.rsi)
                self.contract.candles[self.tf].append(new_candle)

            

            
            self.last_candle = new_candle

            self.parseMutex = False
            return "new_candle"





    def min_max_transform(self,x,min=5000.0,max=100000.0):
        """
                A Min-Max scaling is typically done via the following equation:
                Xsc = (X  - Xmin)/ (Xmax - Xmin).
        """


        return (x - min) / (max-min)

    def min_max_inverse_transform(self,x,min=5000.0,max=100000.0):
        # figure this out it will take a tensor or numpy array 
        """
                A Min-Max scaling is typically done via the following equation:
                X = Xsc *(Xmax - Xmin) +  Xmin    
        """

        return x * (max - min) + min

    def _predict(self):
        """
            This function will make dataframes and predict the next closing 
            We will optimize this for better performance after pipeline is clear    
        """

        # we might not need a data frame simple numpy array will do 
        # convert the closes highs lows opens to a numpy array of the sequence
        
        
        closes = [float(candle.close) for candle in self.contract.candles[self.tf][:-1]]
        highs = [float(candle.high) for candle in self.contract.candles[self.tf][:-1]]
        lows = [float(candle.low) for candle in self.contract.candles[self.tf][:-1]]
        opens = [float(candle.open) for candle in self.contract.candles[self.tf][:-1]]
        last_ts = self.contract.candles[self.tf][-2].timestamp
    
        df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})

        

        X = self.preprocess_data_with_arima(df)
        last_X = X[-1]
        last_X = last_X.reshape(1,last_X.shape[0],last_X.shape[1])

        # Convert the last data point to PyTorch tensor
        last_X = torch.tensor(last_X, dtype=torch.float32)

        outputs,features = [],[]
        with torch.no_grad():
            o, f = self.model(last_X)
            outputs.append(o)
            features.append(f)


        features = torch.cat(features).cpu().numpy()
        predictions = self.regressor.predict(features)
        prediction_upscaled = self.min_max_inverse_transform(predictions)
        self._save_result(prediction_upscaled,last_ts)
        return 

    def _save_result(self,prediction,last_ts):
        logger.info("Prediction for close %s timeframe %s Symbol %s previous candle timestamp %s",prediction.flatten()[0], self.tf, self.contract.symbol,last_ts)
        entry = self.contract.candles[self.tf][-2].close
        pred = prediction.flatten()[0]
        if entry >pred :
            if entry > pred+100:

                exit  = pred +100
            else:
                exit = pred
        # Consider this for the time being save all the predictions 
        elif entry < pred:
            if entry <pred-100:
                exit = pred-100
            else:
                exit = pred
        with open(self.result_csv_path, 'a', newline='') as file:
            file.write(f'{last_ts + self.tf_equiv},{pred},{entry},{exit}\n')




    def check_trade(self, tick_type: str):
        """Checks if there is a new candle and saves prediction named check trade
         to ensure consistency and moving forward this will generate trade signals as well"""
        signal_result = 2
        if tick_type == "new_candle":
            self._predict()
        return signal_result
    

    def preprocess_data_with_arima(self,df, sequence_length = 30):
        # Ensure the dataframe has the required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Dataframe must contain 'open', 'high', 'low', 'close' columns.")
        
        # Extract the relevant data
        data = df[['open', 'high', 'low', 'close']]
        
        # Fit ARIMA model on the close prices and get residuals
        arima_model = ARIMA(df['close'], order=(5, 1, 0))
        arima_fitted = arima_model.fit()
        arima_residuals = arima_fitted.resid
        
        # Fit GARCH model to the returns of the close prices
        returns = df['close'].pct_change().dropna()
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fitted = garch_model.fit(disp='off')
        garch_volatility = garch_fitted.conditional_volatility
        
        # Align the GARCH volatility and ARIMA residuals with the original data
        arima_residuals = arima_residuals.reindex(df.index).fillna(0)
        garch_volatility = garch_volatility.reindex(df.index).fillna(method='bfill').fillna(method='ffill')
        
        # Add ARIMA residuals and GARCH volatility as new features
        data['arima_residuals'] = arima_residuals
        data['garch_volatility'] = garch_volatility

        arima_min = -20000
        arima_max = 20000
        # scalling data with new scaler
        for col in data.columns:
            if col == 'max_arima_residuals':

                data[col] = self.min_max_transform(data[col],arima_min,arima_max)
            else:
                data[col] = self.min_max_transform(data[col])
        
        # scalerX = MinMaxScaler(feature_range=(0, 1))
        dataX  = data.values
                
        # Initialize lists to hold the sequences and corresponding target values
        X= []

        # Loop through the data to create sequences
        for i in range(len(dataX) - sequence_length + 1): # adjusting to include last point and ommit y
            X.append(dataX[i:i + sequence_length])
        
        X = np.array(X)
        
        
        return X
    




