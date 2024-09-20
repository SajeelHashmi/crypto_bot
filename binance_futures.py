# Running with binance instead of testnet 

import logging
import requests
import time
from decimal import Decimal

import typing

from urllib.parse import urlencode

import hmac
import hashlib

import websocket
import json

import threading

from models import *

from strategies import  *

from prediction_models import *

from emailAlerts import sendEmail,sendEmailTest
from datetime import datetime
import asyncio
from queue import Queue
from collections import deque
logger = logging.getLogger()
import traceback
import numpy as np
from telegram_bot import Telegram_Bot
from random_forest  import Random_Forest
from dotenv import load_dotenv
import os
from twilio.rest import Client

load_dotenv()  
# TODO
# Walk Through the whole code and check every where wherever you see a potential error write a log 
# Find bottle necks and refactor in that place
# Identify more bottlenecks 
# close position endpoint Testing 
# telegram bot testing and incorporation 
# Create a strategy controller class that controls strategies fro each coin and time frame right now we are using active contracts dictionary




TF_EQUIV = {"1m": 60, '3m':180,"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400,"1d":86400,"3d":259200,"1w":604800,"1M":2419200}

class BinanceFuturesClient:


    def __init__(self, public_key: str, secret_key: str, testnet: bool):
        if testnet:
            self._base_url = "https://testnet.binancefuture.com"
            self._wss_url = "wss://stream.binancefuture.com/ws"
        else:
            self._base_url = "https://fapi.binance.com"
            self._wss_url = "wss://fstream.binance.com/ws"
        self.stopThreads = False
        self._public_key = public_key
        self._secret_key = secret_key
        account_sid = os.environ.get('TWILLO_SID')
        auth_token = os.environ.get('TWILLO_AUTH')
        self.twillio_client = Client(account_sid,auth_token)

        # print(np.zeros(shape=(1,1)))
        self.telegram = Telegram_Bot(disabled=True)
        logger.info('telegram initialized successfully ')

        self.telegram_active = True

       
       
        
        self._headers = {'X-MBX-APIKEY': self._public_key}
        self._predictionModels = []
        self._models_initialized =False


        self.last_message_time = 0


        self.active_trades :List[Trade] = []
        self.prev_trades:List[Trade] = []
        self.leverage = 5        
        self.trades_allowed = 4
        self.invest_amount = 10
        self.averaging_invest = 7

        logger.info('Getting contracts')

        self.contracts = self.get_contracts()
        logger.info('Contracts retrived')
        candles  = self.get_historical_candles(self.contracts["BTCUSDT"], '1h',limit=1000)
        rsi = self.get_rsi(candles)

        rsi = rsi.iloc[-10:].values
        print(len(rsi))
        for i in range(10):
            candles[-10 + i].rsi = rsi[i]
        
        self.contracts['BTCUSDT'].candles['1h'] = candles
        self.contracts['BTCUSDT'].candlesRetrieved['1h'] = True


        candles  = self.get_historical_candles(self.contracts["BTCUSDT"], '15m',limit=1000)
        rsi = self.get_rsi(candles)

        rsi = rsi.iloc[-10:].values
        print(len(rsi))
        for i in range(10):
            candles[-10 + i].rsi = rsi[i]
        
        self.contracts['BTCUSDT'].candles['15m'] = candles
        self.contracts['BTCUSDT'].candlesRetrieved['15m'] = True
        
        self.random_forest:Random_Forest = Random_Forest(self, self.contracts['BTCUSDT'],  '15m') 
        self.lstm_1h_sub:LSTM_BTC  = LSTM_BTC(self, self.contracts['BTCUSDT'],  '1h')
        logger.info("LSTM 1h -1 INITIALIZED")

        self.lstm_1h_plus:LSTM_BTC  = LSTM_BTC(self, self.contracts['BTCUSDT'],  '1h',plus_1= True)
        logger.info("LSTM 1h +1 INITIALIZED")

        self.lstm_15m:LSTM_BTC  = LSTM_BTC(self,self.contracts['BTCUSDT'],  '15m')
        logger.info("LSTM 15 Min -1 INITIALIZED")


        self.lstm_15m_plus:LSTM_BTC  = LSTM_BTC(self,self.contracts['BTCUSDT'],  '15m',plus_1=True)
        logger.info("LSTM 15 Min +1 INITIALIZED")
        # time.sleep(100)

        
        self.test_symbols =[
                'BTCUSDT',
                'ETHUSDT',
                'BCHUSDT',
                'XRPUSDT',
                'EOSUSDT',
                'LTCUSDT',
                'TRXUSDT',
                'ETCUSDT',
                'XLMUSDT',
                'ADAUSDT',
                'XMRUSDT',
                'DASHUSDT',
                'BNBUSDT',
                'KNCUSDT',
                'DOGEUSDT',
                'SXPUSDT',
                'KAVAUSDT',
                'BANDUSDT',
                'SOLUSDT',
                'KSMUSDT',
                "QTUMUSDT",
                "IOSTUSDT",
                "THETAUSDT",
                "ZILUSDT",
                'MKRUSDT',
                'SNXUSDT',
                'DOTUSDT',
                'DEFIUSDT',
                'YFIUSDT',
                'LINKUSDT'
                ]
        

        self.test_contracts = [self.contracts[x] for x in self.test_symbols]
        # self.balances = self.get_balances()
        self.balances = self.get_balances()
        logger.info("Current USDT BALANCE %s ",self.balances["USDT"].wallet_balance)
        # time.sleep(3)

        # RUNNING ALL STRATEGIES BUT TRADES WILL BE OPEN FOR ONLY FIRST 7 these are optimized strategies which provided the best results 
        self.strategies  = [TweezerBottonStrategy,TweezerTopStrategy,BearishHaramiCross,BullishHaramiCross,PiercingPattern,LongWickBearish,LongWickBullish,BullishEngulfing,BearishEngulfing,HammerStrategy,InvertedHammerStrategy,FallingThree,RisingThree,DarkCloud,ThreeBlackCrows,ThreeWhiteSolidiers,BearishKicker,BullishKicker,MorningStar,EveningStar]
        
        self.stratNames = ['Tweezer Bottom Strategy','Tweezer Top Strategy','Bearish Harami Cross','Bearish Harami Cross','Piercing Pattern','Bullish Engulfing','Bearish Engulfing','Long Wick Bearish','Long Wick Bullish','Hammer','Inverted Hammer','Falling three','Rising three','Dark cloud','Three black crows','Three white solidier','Bearish Kicker','Bullish Kicker','Morning star','Evening Star']
        
        self.daily_polling_running = False
        # self.strategies = [TestStrategy]
        # self.stratNames = ["Test Strategy"]
        
        self.activeContracts: Dict[str,Dict] = dict()
        
        self._ws_id = 1


        self._ws = None
        self.ws_thread = False
        # self.leverageTh = threading.Thread(target=self._set_initial_leverage)
        # self.leverageTh.start()
        
        # for contract in self.contracts.values():
        self.t = threading.Thread(target=self._start_ws)
        self.t.start()


        self.stratId = 0
        self.signalGenerated = False    
        self.noSigMailSent = False

        self.pollingDone = False
        
        

        self.num_of_workers = 50
        self._ws_worker_thread = True
        
        
        MAX_BUFFER_SIZE = 7000
        
        
        self.message_buffer = deque(maxlen=MAX_BUFFER_SIZE)
        logger.info("MESSAGE BUFFER INITIALIZERD")
        
        
        self.buffer_not_empty = threading.Semaphore(0)
        self.buffer_not_full = threading.Semaphore(MAX_BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        
        self.get_open_orders()



        self.workers = []
        for i in range(self.num_of_workers):
            t = Thread(target=self._websocket_worker)
            t.start()
            self.workers.append(t)

        # Thread(target=self._connect_stream).start()

        t = Thread(target=self.no_signal_mail,args=())
        t.start()

        threading.Thread(
                target=self._check_previous_trades
            ).start()



        logger.info("Binance Futures Client successfully initialized")



    def get_balances(self) -> typing.Dict[str, Balance]:
        data = dict()
        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)

        balances = dict()
        
        account_data = self._make_request("GET", "/fapi/v2/account", data)

        if account_data is not None:
            for a in account_data['assets']:
                balances[a['asset']] = Balance(a, "binance")

        return balances

    def get_contracts(self) -> typing.Dict[str, Contract]:
        exchange_info = self._make_request("GET", "/fapi/v1/exchangeInfo", dict())
        contracts = dict()

        if exchange_info is not None:
            for contract_data in exchange_info['symbols']:
                if contract_data['marginAsset'] == "USDT":
                    contracts[contract_data['symbol']] = Contract(contract_data, "binance")

        return contracts

    def get_historical_candles(self, contract: Contract, interval: str,limit = 1000,raw = False) -> typing.List[Candle]:
        data = dict()
        data['symbol'] = contract.symbol
        data['interval'] = interval
        data['limit'] = limit

        raw_candles = self._make_request("GET", "/fapi/v1/klines", data)
        if raw:
            return raw_candles
        candles = []

        if raw_candles is not None:
            for c in raw_candles:
                candles.append(Candle(c, interval, "binance"))
        candles = sorted(candles, key=lambda x: x.timestamp)
        return candles 

    def get_historical_candles_test(self, start ,end,contract: Contract, interval: str,limit = 1000) -> typing.List[Candle]:
        data = dict()
        data['symbol'] = contract.symbol
        data['interval'] = interval
        data['limit'] = limit
        data['startTime'] = start
        data['endTime'] = end

        raw_candles = self._make_request("GET", "/fapi/v1/klines", data)

        candles = []

        if raw_candles is not None:
            for c in raw_candles:
                candles.append(Candle(c, interval, "binance"))

        return candles

    def get_rsi(self,candles):
        period=14
        series = pd.Series([c.close for c in candles])
        delta = series.diff().dropna()

        up, down = delta.copy(), delta.copy()

        up[up < 0] = 0
        down[down > 0] = 0

        avg_gain = up.ewm(com=(period - 1), min_periods=period).mean()
        avg_loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()
        
        rs = avg_gain / avg_loss

        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.round(2)
        return rsi

    def get_open_orders(self):
        data = dict()
        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)

        orders_data = self._make_request("GET","/fapi/v1/openOrders", data)
        sl_tps ={}
        if orders_data is not None:
            for o in orders_data:
                try:
                    sl_tps[o['symbol']]
                except:
                    sl_tps[o['symbol']] = {}
                
                if o['type'] ==  'STOP_MARKET':
                    sl_tps[o['symbol']]['sl'] = o['stopPrice']
                    sl_tps[o['symbol']]['sl_time'] = o['time']                    
                if o['type'] == 'TAKE_PROFIT_MARKET':
                    sl_tps[o['symbol']]['tp'] = o['stopPrice']
                    sl_tps[o['symbol']]['tp_time'] = o['time']
            for key,val  in sl_tps.items():
                if all(key in val for key in ['sl','tp']):
                    print(val)
                    if abs(float(val['sl_time']) - float(val['tp_time']))  > 10000:
                        print("SL TP DIFFERENCE")
                        print(float(val['sl_time']) - float(val['tp_time'])) 
                        continue
                    print(key,val)
                    # We get only active trades sl and tp till here from here we need to check positionRisk api for more infor about trade
                    dataNew = {}
                    dataNew['timestamp'] = int(time.time() * 1000)
                    dataNew['symbol'] = key.lower()
                    dataNew['signature'] = self._generate_signature(dataNew)        
                    position_data = self._make_request("GET", "/fapi/v2/positionRisk", dataNew)
                    trade =Trade({
                                "time": int(time.time() * 1000),
                                'timeframe':'UNKNOWN',
                                "entry_price": position_data[0]['entryPrice'],
                                "contract": self.contracts[key],
                                "strategy": 'UNKNOWN',
                                "side": 'long' if float(position_data[0]['entryPrice']) <float(val['tp']) else 'short',
                                "status": "open",
                                "pnl": 0, 
                                "quantity": '0',
                                "entry_id": '0',
                                "sl_order_id":'0',
                                "tp_order_id":'0',
                                    })
                    
                    print("old trade " ,trade)
                    self.subscribe_channel([self.contracts[key]],"kline_15m")
                    trade.previousTrade = True
                    trade.sl = float (val['sl'])
                    trade.tp = float (val['tp'])
                    self.active_trades.append(trade)
                    self.prev_trades.append(trade)

        return 



    def _check_previous_trades(self):
        while len(self.prev_trades) >0:
            logger.info('checking previous trade')
            for trade in self.prev_trades:
                if trade.status == 'open':

                    symbol = trade.contract.symbol
                    try:
                        position = self.get_open_position(symbol)[0]
                    except:
                        continue
                    if float(position['entryPrice']) ==  0 :
                        result = 'UNKNOWN'
                        trade.status = 'closed'
                        try:
                            if trade.side == 'long':
                                result = 'Take Profit hit' if self.contracts[symbol].candles['15m']['high'] >= trade.tp else 'Stoploss Hit'
                            else:
                                result = 'Take Profit hit' if self.contracts[symbol].candles['15m']['low'] <= trade.tp else 'Stoploss Hit'
                        except:
                            logger.info("could not determine whether stoploss or takeprofit was hit for previous trade")
                            pass

                        msg = f"Trade For Symbol {trade.contract.symbol} Closed\nPNL Unknown (Trade Opened before app start)\n {result}  "
                        logger.info(f"PREVIOUS POSITION CLOSED {msg}")
                        sendEmail(msg , "Old Trade Closed" )
                        self.prev_trades.remove(trade)
                        if trade in self.active_trades:
                            self.active_trades.remove(trade)
                        time.sleep(5)
            time.sleep(40)

    def get_open_position(self,symbol):
        data = {}
        data['symbol'] = symbol        
        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)

        return self._make_request('GET','/fapi/v2/positionRisk',data=data)






    def no_signal_mail(self):
        while True:
            # Get the current time

            current_time = datetime.now().strftime("%H:%M")
            
            # Splitting the current time string into hours and minutes
            hours, minutes = map(int, current_time.split(':'))
            
            
            # Checking if the minutes part is 15, 40, 45, or 0
            if minutes == 1:
                trades_open = [t for t in self.active_trades if t.status =='open'] 
                for trade in trades_open:
                    trade.contract.symbol
                    position = self.get_open_position(trade.contract.symbol)
                    logger.info(f"CHECKING OPEN Trade {trade.id}  {position} ,PNL: {trade.pnl} ,ENTRY: {trade.entry_price} ,SIDE : {trade.side} CURRENT PRICE {self.contracts[trade.contract.symbol].candles['15m'][-1].close }")


                    
                time.sleep(60)

            if minutes > 5 and hours in [1,5,9,13,17,21]:
                self.noSigMailSent = False
                self.signalGenerated  = False

            if minutes in [5] and hours in [1,5,9,13,17,21]:
                if self.signalGenerated :
                    continue
                else:
                    if not self.noSigMailSent:
                        logger.info("No signal was generated")
                        sendEmail(f"No Signal generated within last 4 hours ", subject='No Signal')
                        self.telegram.send_message("No Signal generated within last 4 hours ")
                    self.noSigMailSent = True




            time.sleep(40)

    # def send_telegram_poll(self,msg):
    #     answer_options = [
    #         telebot.types.InputPollOption("YES"),
    #         telebot.types.InputPollOption("NO")
    #     ]
    #     if self.telegram_active:
    #         poll = self.telegramBot.send_poll(
    #         chat_id=self.my_telegram_id,
    #         question=msg +"\n Open this Position?",
    #         options=answer_options,
    #         type="regular",
    #         allows_multiple_answers=False,
    #         is_anonymous=False,
    #         )
    #         return poll.json['poll']['id']
    #     return -1






    
    def startStrategies(self, symbol: str, timeframe: str,  stop_loss_pct: float, take_profit_pct: float,bulk=True):
        print("in start strategies self.ws_thread",self.ws_thread)

        if not self.ws_thread and not bulk:
            return
        key = symbol + '_'+ timeframe
        try:
            self.activeContracts[key]
            return
        except KeyError:
            pass

        obj = {
            "symbol": symbol,
            "timeframe": timeframe,
            'trend':0,
            "strategies": []
        }
        self.stratId += 1
        try:
            try: 
                if not self.contracts[symbol].candlesRetrieved[timeframe]:
                    raise KeyError()
            except KeyError:
                candles  = self.get_historical_candles(self.contracts[symbol], timeframe,limit=200)
                rsi = self.get_rsi(candles)
                rsi = rsi.values
                # print(len(rsi),f"rsi lenght {symbol},{timeframe}")

                for i in range(len(rsi)):
                    candles[(len(rsi)*-1) + i].rsi = rsi[i]

                # print([c.rsi for c in candles[-len(rsi):]])
                if not self.contracts[symbol].candlesRetrieved[timeframe]:
                    self.contracts[symbol].candles[timeframe] = candles
                    self.contracts[symbol].candlesRetrieved[timeframe] = True
                # logger.info("Symbol %s has RSI %s on time frame %s while initailizing" , symbol,rsi[-1],timeframe )
            for strat in self.strategies:
                newStrat = strat(self, self.contracts[symbol], "Binance", timeframe, self.invest_amount,take_profit_pct , stop_loss_pct)

                obj["strategies"].append({
                    "strategy": newStrat,
                    "status": "running",
                    'signal':0
                })
            self.activeContracts[key] = obj
            # if symbol == 'BTCUSDT' and not self._models_initialized:
            #     self._predictionModels = [Lstm_model(self, self.contracts['BTCUSDT'], "Binance", '1h', 1, 1, 1),GRU_model(self, self.contracts['BTCUSDT'], "Binance", '1h', 1, 1, 1)]
                # self._models_initialized =True

        except Exception as e:
            print("Error while starting the strategy" , traceback.format_exc())
        if bulk:
            return
        if self.ws_thread:
            self.subscribe_channel([self.contracts[symbol]], f"kline_{timeframe}")

    def stop_strategy(self, key:str):
        self.unsubscribe_channel([self.contracts[self.activeContracts[key]['symbol']]], "aggTrade")
        try:
            del self.activeContracts[key]
        except KeyError as e:
            return "failed"
        return "success"
    
    def start_test_strats(self):
        # self.subscribe_channel(self.test_contracts, "aggTrade")
        # switching off monthly websockets to decrease some load
        
        timeframes= [ "15m","30m","1h", "4h",'1d','3d','1w','1M'
                     ]
            
        for s in self.test_symbols:
            for tf in timeframes:    
                self.startStrategies(s,tf,5,1)
        timeframes= [ "15m","30m" ,"1h", "4h"
                    #  ,'1d','3d'
                    #  '1w',
                    #  '1M'
                     ]
        for tf in timeframes:
            self.subscribe_channel(self.test_contracts,f"kline_{tf}")
            time.sleep(1)
        
        Thread(target=self._daily_polling).start()


    def stop_test_strats(self):
        self.unsubscribe_channel(self.test_contracts, "aggTrade")
        timeframes= [ "15m","30m","1h", "4h",'1d','3d','1w','1M']
        for tf in timeframes:
            self.unsubscribe_channel(self.test_contracts,f"kline_{tf}")
        for s in self.test_symbols:
            for tf in timeframes:
                while 1:
                    if not self.ws_thread:
                        time.sleep(50)
                    break
                try:
                    self.stop_strategy(f"{s}_{tf}")
                    time.sleep(1)
                except  KeyError as e:
                    print(f"{e} does not exists in keys")
                    continue





    def _generate_signature(self, data: typing.Dict) -> str:
        return hmac.new(self._secret_key.encode(), urlencode(data).encode(), hashlib.sha256).hexdigest()

    def _make_request(self, method: str, endpoint: str, data: typing.Dict):
        if method == "GET":
            try:
                response = requests.get(self._base_url + endpoint, params=data, headers=self._headers)
            except Exception as e:
                logger.error("Connection error while making %s request to %s: %s", method, endpoint, e)
                return None

        elif method == "POST":
            try:
                response = requests.post(self._base_url + endpoint, params=data, headers=self._headers)
            except Exception as e:
                logger.error("Connection error while making %s request to %s: %s", method, endpoint, e)
                return None

        elif method == "DELETE":
            try:
                response = requests.delete(self._base_url + endpoint, params=data, headers=self._headers)
            except Exception as e:
                logger.error("Connection error while making %s request to %s: %s", method, endpoint, e)
                return None
        elif method == "PUT":
            try:
                response = requests.put(self._base_url + endpoint, params=data, headers=self._headers)
            except Exception as e:
                logger.error("Connection error while making %s request to %s: %s", method, endpoint, e)
                return None
        else:
            raise ValueError()
        if response.status_code == 200:
            return response.json()
        else:
            logger.error("Error while making %s request to %s: %s (error code %s)",
                         method, endpoint, response.json(), response.status_code)
            if endpoint == '/fapi/v1/order' and method == "POST":
                sendEmail(f'COULD NOT OPEN POSITION \n ERROR: {response.json()}')
            logger.error(f'data for failed req {data}')
            return None





    def _daily_polling(self):
        """Instead of wasting websocket resources on long time frames this will poll for all the candles daily 3 days weekly and monthly candles every night at 5am """
        if self.daily_polling_running :
            logger.info("daily polling already running")
            return
        self.daily_polling_running =True
        logger.info("DAILY POLLING HAS STARTED")
        while True:
            current_time = datetime.now().strftime("%H:%M")
            
            # Splitting the current time string into hours and minutes
            hours, minutes = map(int, current_time.split(':'))
            # print(hours,minutes)

            if hours == 5 and minutes <= 4:

                for tf in ['1d','3d','1w','1M']:
                    for contract  in self.test_contracts:
                        data = self.get_historical_candles(contract,tf,limit=1,raw=True)[0]
                        candle = {
                                's':contract.symbol,
                            'k':{
                                't': data[0],
                                's':contract.symbol,
                                'i':tf,
                                "o": float(data[1]),
                                "h": float(data[2]),
                                "l": float(data[3]),
                                "c": float(data[4]),
                                "v": float(data[5]),
                                
                            }
                        }
                        logger.info("Candle Recievied while daily Polling %s Symbol %s timeframe %s",candle['k']['t'] ,contract.symbol,tf)


                        self._parse_msg_ws(candle)

            time.sleep(115)

    def _start_polling(self):
        logger.info("switching To polling while waiting for websocket to reconnect")
        while not self.ws_thread:
            try:
                for key,row in self.activeContracts.items():
                        if self.ws_thread:
                            logger.info("Websocket Reconnected Stopping Polling and switching back to ws stream")
                            self.pollingDone = True
                            return
                        symbol = key.split('_')[0]
                        contract = self.contracts[symbol]
                        timeframe = key.split('_')[1]
                        data = self.get_historical_candles(contract,timeframe,limit=1,raw=True)[0]
                        candle = {
                            's':contract.symbol,
                           'k':{
                               't': data[0],
                               's':symbol,
                               'i':timeframe,

                               "o": float(data[1]),
                               "h": float(data[2]),
                               "l": float(data[3]),
                               "c": float(data[4]),
                               "v": float(data[5]),

                           }
                       }
                        logger.info("Candle Recievied while polling %s Symbol %s timeframe %s",candle['k']['t'] ,symbol,timeframe)
                        self._parse_msg_ws(candle)
                        

            except Exception as e:
                logger.info(f"Error while Polling : {e}")
                continue
        logger.info("Websocket Reconnected Stopping Polling and switching back to ws stream")
        self.pollingDone = True

    def _start_ws(self,reconnect=False):
        self._ws = websocket.WebSocketApp(self._wss_url, on_open=self._on_open, on_close=self._on_close,on_error=self._on_error, on_message=self._on_message)
        try:
            self.ws_thread =True
            self._ws.run_forever()
        except Exception as e:
            logger.error("Binance error in run_forever() method: %s", e)

    def _on_open(self, ws):
        logger.info("Binance connection opened")
        print(len(self.test_contracts))
        print( [self.contracts['BTCUSDT']])
        self.subscribe_channel([self.contracts['BTCUSDT']],f"kline_1h")           
        self.subscribe_channel([self.contracts['BTCUSDT']],f"kline_15m")           

    def _on_close(self, ws,*args, **kwargs):
        self.ws_thread =False
        time.sleep(10)
        logger.warning("Binance Websocket connection closed unprocessed message : %s",len(self.message_buffer))
    
        # sleep for 3 minutes no matter what
        try:
            logger.info("Waiting 3 minutes before reconnecting")
            self.message_buffer.clear()
            # shift to polling in this time frame using threades and when wait is over and ws_starts again shift back to websockets
            Thread(target=self._start_polling).start()
            time.sleep(180) 
            logger.info("Wait over.")
        except Exception as e:
                    logger.error("An error occurred during the sleep period: %s", e)
        logger.info("Close reason: %s", kwargs)
        if args:
            logger.info("Close reason: %s", args[0])
            logger.info("Args: %s", args)  # Log the entire args list for inspection
            if args[0] == '1008' or args[0] == 1008:
                try:
                    logger.info("Waiting 3 minutes...")
                    time.sleep(180) 
                    logger.info("Wait over.")
                except Exception as e:
                    logger.error("An error occurred during the sleep period: %s", e)
            else:
                logger.info("No need to wait. Continuing...")

        logger.info("Reconnecting......")
  
        self.t = threading.Thread(target=self._start_ws)
        self.t.start()
        while 1:
            time.sleep(5)
            if self.ws_thread:
                self.start_test_strats()
                time.sleep(5)
                break

    




    def _on_error(self, ws, msg: str):
        logger.error("Binance connection error: %s", msg)
        
    def _websocket_worker(self):
        while True:
            try:
                self.buffer_not_empty.acquire()
                data = self.message_buffer.popleft()

                self._parse_msg_ws(data)

                 
            except IndexError:

                continue

            except Exception as e:
                logger.error(f"Error processing message in websokcet worker: {e}")
                            
    def _on_message(self, ws, msg: str):
        try:
            data = json.loads(msg)
            
            if "e" in data and data['e'] == "kline":
                
                if len(self.message_buffer )> 6500:
                    self.message_buffer.clear()
                    logger.info("clearing buffer")

                
                # if self.buffer_lock.acquire(False):
                self.message_buffer.append(data)
                self.buffer_not_empty.release()  # Signal that buffer is not empty
                # self.buffer_lock.release()

        except Exception as e:
            logger.info(f"Error processing message in on message handler: {e}")
                   
    def check_trades(self,strat:Strategy,data):
        try:
            strat.check_active_trades(data)

        except Exception as e:
            logger.error("Erorr While Checking Trades ",e)

    def check_signal(self,strat,res:str):
        
        try:

            signal = strat['strategy'].check_trade(res)
            
            if strat['signal'] is None:
                strat['signal'] = 0
            if signal in [1,-1]:
                self.signalGenerated = True
            strat['signal'] = strat['signal'] if  signal == 2 else signal
        except Exception as e:
            logger.info("Erorr While Checking Signal ",e)

    def _parse_msg_ws(self,data):
        """Loops over strategies and calls parse_trade and check signal using threading for relevant symbol and timeframe """
        symbol = data['s']
        timeframe = data['k']['i']
  
        
        try:
            for key,row in self.activeContracts.items():
                    s = key.split('_')[0]
                    tf = key.split('_')[1]
                    if s == symbol and tf == timeframe:
                        
                        res = self.parse_candle(data['k'])
                        for strat in row['strategies']:
                            
                            row['trend'] = row['strategies'][0]['strategy'].find_trend()
                            self.check_trades(strat['strategy'],data)
                            try:
                                self.check_signal(strat,res)
                            except Exception as e:
                                logger.info(f"exception for {strat['strategy'].contract.symbol} While parsing trades , {e}")
                                break
                        if symbol == 'BTCUSDT' and res=='new_candle':

                            Thread(
                                target=self._check_predictions,
                                args=(res,timeframe)
                                ).start()
                        break

        except Exception as e:
            logger.info(f"Error while looping through the Binance strategies Kline Channel: {e}")

    def _update_rsi(self,symbol,tf):
        period=14
        series = pd.Series([c.close for c in self.contracts[symbol].candles[tf]])
        delta = series.diff().dropna()

        up, down = delta.copy(), delta.copy()

        up[up < 0] = 0
        down[down > 0] = 0

        avg_gain = up.ewm(com=(period - 1), min_periods=period).mean()
        avg_loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()
    
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.round(2)

        rsi = rsi.values[-1]
        self.contracts[symbol].candles[tf][-1].rsi = rsi
        # logger.info("NEW RSI FOR %s is %s",symbol,self.contracts[symbol].candles[tf][-1].rsi)
        
        return rsi

    def parse_candle(self,data):
        """Parsing Candle to see if new candle has appeared or not"""
        # Check if a new candle is recieved or not Previously we were doing this in strategies class but that leads to every strategy doing same calcultaion

        

        tf = data['i']
        symbol = data['s']
        last_candle = self.contracts[symbol].candles[tf][-1]
    
        tf_equiv = TF_EQUIV[tf] * 1000
        # print(data)
        timestamp_kline = float(data['t'])

        # SAME CANDLE
        if timestamp_kline < self.contracts[symbol].candles[tf][-1].timestamp + tf_equiv:

            self.contracts[symbol].candles[tf][-1].close = float(data['c'])
            self.contracts[symbol].candles[tf][-1].high = float(data['h'])
            self.contracts[symbol].candles[tf][-1].low = float(data['l'])
            self.contracts[symbol].candles[tf][-1].volume = float(data['v'])
            

            # Check Take profit / Stop loss



            self._update_rsi(symbol,tf)
            return "same_candle"

        # Missing Candle(s)

        elif timestamp_kline >=  self.contracts[symbol].candles[tf][-1].timestamp + 2 * tf_equiv:

            missing_candles = int((timestamp_kline - last_candle.timestamp) / tf_equiv) - 1

            logger.info("Missing %s candles for %s %s (%s %s)",  missing_candles, self.contracts[symbol],
                        tf, timestamp_kline, last_candle.timestamp)

            candles  = self.get_historical_candles(self.contracts[symbol], tf,limit=1000)

            rsi = self.get_rsi(candles)

            rsi = rsi.iloc[-10:].values
            print(len(rsi))
            for i in range(10):
                candles[-10 + i].rsi = rsi[i]


            self.contracts[symbol].candles[tf] = candles
            
            return "new_candle"

        # New Candle
        elif timestamp_kline >=  self.contracts[symbol].candles[tf][-1].timestamp + tf_equiv:
            
            new_ts = timestamp_kline
            candle_info = {'ts': new_ts, 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['v']}
            new_candle = Candle(candle_info, tf, "parse_trade")
            
            self.contracts[symbol].append_candle(new_candle,tf)

            logger.info("New candle for %s %s Latest Candle Timestamp %s ,RSI %s New Candle Timestamp %s, New Candle RSI %s High %s Low %s Close %s ",
                            symbol, 
                            tf,self.contracts[symbol].candles[tf][-2].timestamp ,
                            self.contracts[symbol].candles[tf][-2].rsi , 
                            new_candle.timestamp , 
                            new_candle.rsi,
                            self.contracts[symbol].candles[tf][-2].high , 
                            self.contracts[symbol].candles[tf][-2].low ,
                            self.contracts[symbol].candles[tf][-2].close)
            
            self.last_candle = new_candle
            self._update_rsi(symbol,tf)
            return "new_candle"
        pass

    def subscribe_channel(self, contracts: typing.List[Contract], channel: str):
        print("subscribing channel")
        data = dict()
        data['method'] = "SUBSCRIBE"
        data['params'] = []
        for contract in contracts:
            data['params'].append(contract.symbol.lower() + "@" + channel)
        data['id'] = self._ws_id
        try:
            self._ws.send(json.dumps(data))

        except Exception as e:
            if self.ws_thread:
                logger.error("Websocket error while subscribing to %s %s updates: %s", len(contracts), channel, e)
                time.sleep(2)
                self.subscribe_channel(contracts, channel)
            else:
                logger.error("Websocket Not initialized %s %s updates: %s", len(contracts), channel, e)

        self._ws_id += 1
        print("subscribed ",self._ws_id ,f" chanel {channel}")
    
    def unsubscribe_channel(self, contracts: typing.List[Contract], channel: str):
        data = dict()
        data['method'] = "UNSUBSCRIBE"
        data['params'] = []
        for contract in contracts:
            data['params'].append(contract.symbol.lower() + "@" + channel)
        data['id'] = self._ws_id
        try:
            self._ws.send(json.dumps(data))
            print(f"unsubed channels {len(contracts)}")

        except Exception as e:
            if self.ws_thread:
                logger.error("Websocket error while subscribing to %s %s updates: %s", len(contracts), channel, e)
                time.sleep(2)
                self.unsubscribe_channel(contracts, channel)
            else:
                logger.error("Websocket Not initialized %s %s updates: %s", len(contracts), channel, e)

        try:
            self._ws.send(json.dumps(data))

        except Exception as e:
            logger.error("Websocket error while unsubscribing to %s %s updates: %s", contract, channel, e)

        self._ws_id += 1

    def _check_predictions(self,res,timeframe):
        if timeframe =='1h' : 
            self.lstm_1h_plus.check_trade(res)
            self.lstm_1h_sub.check_trade(res)

        elif timeframe =='15m':
            self.lstm_15m.check_trade(res)
            self.lstm_15m_plus.check_trade(res)
            self.random_forest.check_trade(res)




    
    def get_trade_size(self, contract: Contract, price: float, amount: float,leverage =None):
        if leverage is None:
            leverage = self.leverage
        balance = self.get_balances()
        if balance is not None:
            if 'USDT' in balance:
                balance = balance['USDT'].wallet_balance
            else:
                return None
        else:
            return None
        if balance<= amount:
            return None
        

        trade_size = (amount * leverage / price) 

        print(trade_size)
        trade_size = round(round(trade_size / contract.lot_size) * contract.lot_size, 8)
        logger.info("Binance Futures current USDT balance = %s, trade size = %s", balance, trade_size)

        return trade_size

    def place_order(self, contract: Contract, order_type: str, quantity: float, side: str, last_price=None, tif=None,tp = None,sl=None,absolute_tp=None) -> Tuple[OrderStatus,float]:
        " add a return of stop loss price and takeprofit pice "
        data = dict()
        data['symbol'] = contract.symbol
        data['side'] = side.upper()
        data['quantity'] = quantity
        data['type'] = order_type
        ## Can Be any candle we just want the last price

   
        # print(stop_price)
        if order_type=="TAKE_PROFIT_MARKET":

            if side == 'buy':
                if absolute_tp:
                    takeprofit = absolute_tp
                else:
                    takeprofit = last_price +   (last_price * tp/100)
                
                data['side'] = "SELL"
            elif side =='sell':
                if absolute_tp:
                    takeprofit = absolute_tp
                else:
                    takeprofit = last_price -   (last_price * tp/100)


                data['side'] = "BUY"
            data["stopPrice"] = round(round(takeprofit / contract.tick_size) * contract.tick_size, 8)
            logger.info(f'takeprofit pct  = {tp} takeprofit = {data["stopPrice"]}')
            data['closePosition'] = True


        if order_type=="STOP_MARKET":
            if side == 'buy':
                data['side'] = "SELL"
                stoploss = last_price -   (last_price * sl/100)
            elif side =='sell':
                data['side'] = "BUY"
                stoploss = last_price +   (last_price * sl/100)
            data["stopPrice"] = round(round(stoploss / contract.tick_size) * contract.tick_size, 8)
            logger.info(f'stoploss pct  = {sl} stoploss = {data["stopPrice"]}')
            data['closePosition'] = True

        # print(price)
        # if price is not None:
        #     data['price'] = round(round(price / contract.tick_size) * contract.tick_size, 8)

        if tif is not None:
            data['timeInForce'] = tif

        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)

        order_status = self._make_request("POST", "/fapi/v1/order", data)
        logger.info("ORDER STATUS %s ", order_status)

        if order_status is not None:
            order_status = OrderStatus(order_status, "binance")

            stopPrice = 0.0
            try:
                stopPrice = data['stopPrice']
            except:
                pass
            return order_status,stopPrice  
        return None,None
    
    def round_step_size(self,quantity: Union[float, Decimal], step_size: Union[float, Decimal]) -> float:
        """Rounds a given quantity to a specific step size

        :param quantity: required
        :param step_size: required

        :return: decimal
        """
        quantity = Decimal(str(quantity))
        return float(quantity - quantity % Decimal(str(step_size)))

    def modify_order(self,orderId, contract: Contract, order_type: str, quantity: float, side: str,takeprofit:float) -> OrderStatus:
        data = dict()
        data['orderId'] = orderId
        data['symbol'] = contract.symbol
        data['side'] = side.upper()
        data['quantity'] = quantity
        data['type'] = order_type
        ## Modify this to match calculations for new takeprofit and stoploss
        if order_type=="TAKE_PROFIT_MARKET":
            if side == 'buy':
                data['side'] = "SELL"
            elif side =='sell':
                data['side'] = "BUY"


# round(round(takeprofit / contract.tick_size) * contract.tick_size, 8)
            data["stopPrice"] = round(round(takeprofit / contract.tick_size) * contract.tick_size, 8)
            logger.info(f'NEW takeprofit For symbol {contract.symbol} set at  {data["stopPrice"]}')
        else: 
            return

        data['price'] = round(round(takeprofit / contract.tick_size) * contract.tick_size, 8)



        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)

        order_status = self._make_request("PUT", "/fapi/v1/order", data)
        logger.info(f"ORDER STATUS AFTER MODIFYING {order_status}" )

        if order_status is not None:
            order_status = OrderStatus(order_status, "binance")

        return order_status

    def cancel_order(self, contract: Contract, order_id: int) -> OrderStatus:

        data = dict()
        data['orderId'] = order_id
        data['symbol'] = contract.symbol

        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)

        order_status = self._make_request("DELETE", "/fapi/v1/order", data)

        if order_status is not None:
            order_status = OrderStatus(order_status, "binance")

        return order_status

    def get_order_status(self, contract: Contract, order_id: int) -> OrderStatus:

        data = dict()
        data['timestamp'] = int(time.time() * 1000)
        data['symbol'] = contract.symbol
        data['orderId'] = order_id
        data['signature'] = self._generate_signature(data)

        order_status = self._make_request("GET", "/fapi/v1/order", data)
        logger.info("ORDER STATUS IN CHECK ORDER STATUS %s ", order_status)

        if order_status is not None:
            order_status = OrderStatus(order_status, "binance")

        return order_status

    def close_position(self,symbol,stat_name):
        logger.info('close position called for %s for strategy %s',symbol,stat_name)
        try:
            for key,row in self.activeContracts.items():
                for strat in row['strategies']:

                    if strat['strategy'].contract.symbol.lower() == symbol.lower() and strat['strategy'].stat_name.lower() == stat_name.lower():
                        trade = strat['strategy'].trades[-1]
                        msg = strat['strategy'].close_position(trade)
                        return msg
            return "no such active trade"
                                
        except IndexError as e:
            logger.info(f"Strategies not initialized for {strat['strategy'].contract.symbol}")  
        except Exception as e:
            logger.info(f"Error while looping through the Binance strategies: {e}")

    def _set_initial_leverage(self):
        for contract in self.contracts.values():
            data = dict()
            data['symbol'] = contract.symbol
            data['leverage'] = self.leverage
            
            data['timestamp'] = int(time.time() * 1000)
            data['signature'] = self._generate_signature(data)

            res = self._make_request("POST", "/fapi/v1/leverage", data)
            
            if res is not None:
                logger.info(f"Initial Leverage set to 5 for {contract.symbol}")
               
    def change_leverage(self,leverage,contract):
        data = dict()
        data['symbol'] = contract.symbol
        data['leverage'] = leverage
        
        data['timestamp'] = int(time.time() * 1000)
        data['signature'] = self._generate_signature(data)
        res = self._make_request("POST", "/fapi/v1/leverage", data)
        
        if res is not None:
                logger.info(f"Leverage set to {leverage} for {contract.symbol}")
            

