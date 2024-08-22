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

import os 
import csv

from arch import arch_model
import torch
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger()
TF_EQUIV = {"1m": 60, '3m':180,"5m": 300, "15m": 900, "30m": 900, "1h": 3600, "4h": 14400,"1d":86400,"3d":259200,"1w":604800,"1M":2592000}



class Random_Forest():
    def __init__(self, client, contract: Contract, timeframe: str):
        print("initializing RANDOM FOREST CLASSIFIER")
        if not os.path.exists('predictionsResults'):
            os.makedirs('predictionsResults')
        self.result_csv_path = f"predictionsResults/random_forest_{timeframe}.csv"
        
        self.client = client
        self.contract = contract
        self.tf = timeframe
        self.tf_equiv = TF_EQUIV[timeframe] * 1000
        self.trades = []


        with open('saved_models/random_forest.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        self.exchange = 'Binance'
        headers = ['timestamp','prediction','entry']
        if not os.path.exists(self.result_csv_path):
            # Open the file in write mode
            with open(self.result_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write the headers
                writer.writerow(headers)            
        # Load the saved model

        if self.contract.candlesRetrieved[self.tf]:
            self.last_candle = self.contract.candles[self.tf][-1]


        self.contract = contract


        self.parseMutex = False
        self.predictMutex = False

        self._predict()
    






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
        vols = [float(candle.volume) for candle in self.contract.candles[self.tf][:-1]]
        rsis = [float(candle.rsi) for candle in self.contract.candles[self.tf][:-1]]
        
        last_ts = self.contract.candles[self.tf][-2].timestamp
    
        df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes,'volume':vols,'rsi':rsis})

        

        X = self.preprocess_data_with_arima(df)
        last_X = X[-1]
        last_X = last_X.reshape(1,-1)
        prediction = self.model.predict(last_X)
        self._save_result(prediction,last_ts)
        return 

    def _save_result(self,prediction,last_ts):
        if prediction ==0:
            prediction = 'UP'

        elif prediction == 1:
            prediction = 'DOWN'
        
        else:
            prediction = "NEUTRAL"
        logger.info("Prediction for close %s timeframe %s Symbol %s previous candle timestamp %s",prediction, self.tf, self.contract.symbol,last_ts)
        entry = self.contract.candles[self.tf][-2].close
        
  
        with open(self.result_csv_path, 'a', newline='') as file:
            file.write(f'{last_ts + self.tf_equiv},{prediction},{entry}\n')




    def check_trade(self, tick_type: str):
        """Checks if there is a new candle and saves prediction named check trade
         to ensure consistency and moving forward this will generate trade signals as well"""
        signal_result = 2
        if tick_type == "new_candle":
            self._predict()
        return signal_result
    
    def preprocess_data_with_arima(self,df, sequence_length=5):
        # Ensure the dataframe has the required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Dataframe must contain 'open', 'high', 'low', 'close' columns.")
        
        # Extract the relevant data
        data = df[['open', 'high', 'low', 'close' ,'volume','rsi']]


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

        
        arima_min = -20000.0
        arima_max = 20000.0

        rsi_max= 100.0
        rsi_min = 0.0

        vol_min = 100.0
        vol_max = 180000

        for col in data.columns:
            if col =='rsi':
                data[col] = self.min_max_transform(data[col],rsi_min,rsi_max)
            elif col =='volume':
                data[col] = self.min_max_transform(data[col],vol_min,vol_max)
            elif col == 'arima_residuals':
                data[col] = self.min_max_transform(data[col],arima_min,arima_max)
            else:
                data[col] = self.min_max_transform(data[col])
        
        dataX  = data.values
        

        X,y =  [],[]

        # Loop through the data to create sequences
        for i in range(len(dataX) - sequence_length  ):

            # considering i = 0 elements 0:30 30 not included
            X.append( dataX[i:i + sequence_length].flatten() )

        # Convert lists to numpy arrays
        X = np.array(X)
 
        return X