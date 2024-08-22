import logging
from typing import *
import time

from threading import Timer,Thread,Lock

import pandas as pd

from models import *

from emailAlerts import sendEmail,sendEmailTest
import csv
from datetime import datetime


# from connection import Database


if TYPE_CHECKING:
    from binance_futures import BinanceFuturesClient

logger = logging.getLogger()


import numpy as np
import requests


# TODO
# AVERAGING WORKING STILL KEEP TESTING
# REFACTOR AND OPTIMIZE
# UPDATE TRADES TRACKING HOW MANY POSITIONS ARE OPEN ETC
# USE POSITION RISK ENDPOINT TO DETERMINE WETHER POSITION IS OPEN OR NOT



TF_EQUIV = {"1m": 60, '3m':180,"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400,"1d":86400,"3d":259200,"1w":604800,"1M":2419200}


"""Divide strategies on bullish and bearish signals which will only signal a long or short trade"""

class Strategy:
    def __init__(self, client, contract: Contract, exchange: str,
                 timeframe: str, trade_amount: float, take_profit: float, stop_loss: float,strat_name,signal_type = 'long',averaging_strat = False):
        self.averaging_strat = averaging_strat
        self._signal_type = signal_type
        self.client:BinanceFuturesClient = client
 
        self.contract = contract
        self.exchange = exchange
        self.tf = timeframe
        self.tf_equiv = TF_EQUIV[timeframe] * 1000
        if self.averaging_strat:
            self.trade_amount = self.client.averaging_invest
        else:
            self.trade_amount = trade_amount
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.poll_id = None

        self.stat_name = strat_name

        self.parseMutex = False
        self.tradeMutex = Lock()
        self.averagingMutex = Lock()

        self.leverage = self.client.leverage
        # TODO
        ## Create a data structure to store signals generated 
        ## The signals must include stoploss and takeprofit along side entry position 
        ## When the user returns yes from poll the relevant trade must be spotted and taken 

        self.signals:Dict[str,Signal] = {}


        # this history_csv will
        self.history_csv = fr"C:\Users\Asad\Desktop\tradingBotWeb\app\history\{self.stat_name}_history.csv" 
        self.history_dict = []

        # headers = [
        #         'symbol',
        #         'strategy',
        #         'orderSide',
        #         'timeframe',
        #         'timestamp',
        #         'takeprofit_pct',
        #         'stoploss_pct',
        # ]

        # with open(self.history_csv,'w') as file:
        #     for h in headers:
        #         file.write(f"{h},")
        #     file.write('\n') 

        self.ongoing_position = False

        self.sl_tp_mutex = Lock()

        self.trend = self.find_trend()
        self.trades: List[Trade] = []
        if self.contract.candlesRetrieved[self.tf]:
            self.last_candle = self.contract.candles[self.tf][-1]
        else:
            print("candles not recieved for last candle ")
   
    def check_trade(res:str) -> int:
        pass
    
    def find_trend_test(self,start,end,changePct = 0.15):
        if start -  abs(start * changePct) > end :
            return -1
        elif end  > start + abs(start * changePct):
            return 1
        return 0
    
    def _update_rsi(self):
        period=14
        series = pd.Series([c.close for c in self.contract.candles[self.tf]])
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
        self.contract.candles[self.tf][-1].rsi = rsi

        
        return rsi

    def find_trend(self,bearish = 20.0,bullish = 80.0):
        # check moving average of volume also
        candles = self.contract.candles[self.tf][-17:]  
        vol = [c.volume for c in candles]
        df = pd.DataFrame({'volume': vol})

        volMa = df['volume'].ewm(span=14, adjust=False).mean().iloc[-3]

        if volMa >= self.contract.candles[self.tf][-3].volume:
            return 0 
        if self.contract.candles[self.tf][-3].rsi == 0 or self.contract.candles[self.tf][-3].rsi ==100:
            return 0
        if self.contract.candles[self.tf][-3].rsi > bullish  :
            # logger.info(f"Trending Market for symbol {self.contract.symbol} RSI {self.contract.candles[self.tf][-2].rsi} timeframe {self.tf}" )

            return 1
        elif self.contract.candles[self.tf][-3].rsi < bearish:
            # logger.info(f"Trending Market for symbol {self.contract.symbol} RSI {self.contract.candles[self.tf][-2].rsi} timeframe {self.tf}" )
            return -1
        return 0
 

    def send_telegram_msg(self,msg):
        try:
            self.client.telegram.send_message(msg)
        except:
            logger.info('could not send telegram msg')



    def send_msg(self,msg:str,add_info = True,openPos=None,trade =False):
        if not trade:
            tradeId = f'{self.stat_name}_{self.contract.symbol}_{self.tf}'
            Thread(target=self.send_telegram_msg,args=(msg,)).start()
            Thread(target=sendEmailTest,args=(msg,tradeId)).start()

            return


        if add_info:
            if not openPos: 
                openPos = self.last_candle.close
            msg += f"\nOpen Position : {openPos}\nSide : {self._signal_type}"
            if self._signal_type =='long':
                takeProfit = openPos * (1 + self.take_profit / 100)
                stopLoss = openPos * (1 - self.stop_loss / 100)

            else:
                takeProfit = openPos * (1 - self.take_profit / 100)
                stopLoss = openPos * (1 + self.stop_loss / 100)
            msg += f"\nTP: {takeProfit} ({self.take_profit}%)\nSL : {stopLoss} ({self.stop_loss}%)\nInvestment Amount : $ {self.client.invest_amount}\nLeverage : {self.client.leverage}X"


        logger.info("Sending message %s ",msg)
        if self.stat_name == 'TEST':
            Thread(target=sendEmailTest,args=(msg,)).start()
            Thread(target=self.send_telegram_msg,args=(msg,)).start()

            return
        Thread(target=sendEmail,args=(msg,)).start()
        Thread(target=self.send_telegram_msg,args=(msg,)).start()

        return
        self.send_msg_telegram(msg)
        message = self.twillio_client.messages.create(from_ = '+18669859293',body=msg,to='+923139534555')
        print(message.sid)
                





    def check_active_trades(self,data):
        self._check_tp_sl(data)

    def parse_trades(self, data) -> str:
        if self.parseMutex:
            return
        self.parseMutex = True


        if self.tf != data['i']:
            self.parseMutex = False
            return "Different TF"

        last_candle = self.last_candle


        timestamp_kline = float(data['t'])


   
        # SAME CANDLE
        if timestamp_kline < last_candle.timestamp + self.tf_equiv:

            last_candle.close = float(data['c'])
            last_candle.high = float(data['h'])
            last_candle.low = float(data['l'])

            # Check Take profit / Stop loss

            for trade in self.trades:
                if trade.status == "open" and trade.entry_price is not None:     
                    self._check_tp_sl(trade)
                    self.averaging()
             


            self.parseMutex = False

            return "same_candle"

        # Missing Candle(s)

        elif timestamp_kline >= last_candle.timestamp + 2 * self.tf_equiv:

            missing_candles = int((timestamp_kline - last_candle.timestamp) / self.tf_equiv) - 1

            logger.info("%s missing %s candles for %s %s (%s %s)", self.exchange, missing_candles, self.contract.symbol,
                        self.tf, timestamp_kline, last_candle.timestamp)

            candles  = self.client.get_historical_candles(self.contract, self.tf,limit=1000)

            Thread(target=self._update_rsi).start()




            self.contract.candles[self.tf] = candles
            self.last_candle =  self.contract.candles[self.tf][-1]
            self.parseMutex = False
            return "new_candle"

        # New Candle
        elif timestamp_kline >= last_candle.timestamp + self.tf_equiv:
            
            new_ts = timestamp_kline
            candle_info = {'ts': new_ts, 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['v']}
            new_candle = Candle(candle_info, self.tf, "parse_trade")
            
            if timestamp_kline >= self.contract.candles[self.tf][-1].timestamp + self.tf_equiv:
                logger.info("%s New candle for %s %s Latest Candle Timestamp %s ,RSI %s New Candle Timestamp %s, New Candle RSI %s", self.exchange, self.contract.symbol, self.tf,self.contract.candles[self.tf][-1].timestamp,self.contract.candles[self.tf][-1].rsi,new_candle.timestamp,new_candle.rsi)
                self.contract.candles[self.tf].append(new_candle)

                 

            

            
            self.last_candle = new_candle
            self.parseMutex = False
            return "new_candle"

    def _check_order_status(self, order_id):
        print("\n\n\nCHECKING ORDER STATUS\n\n\n\n\n")
        order_status = self.client.get_order_status(self.contract, order_id)
        if order_status is not None:

            logger.info("%s order status: %s Order id %s", self.exchange, order_status.status,order_id)

   
        return order_status

    def averaging(self):

        
        if not self.averagingMutex.acquire(False):
            return
       
        if not self.averaging_strat:
            return 
        
        for t in self.trades:
        
            if t.status != 'open':
                continue
            
            
            entry = t.entry_price


            if t.averaged:
                self.averagingMutex.release()
                return
            

            if t.side == 'long':
                
                if self.contract.candles[self.tf][-1].close <= entry - (entry*0.01):

                    trade_size = self.client.get_trade_size(self.contract, self.contract.candles[self.tf][-1].close, self.trade_amount , leverage=self.leverage)
        
                    logger.info("AVERAGING REQUIREMENT FULFILED FOR SYMBOL %s ON TIMEFRAME %s TRADE SIZE TO EXTEND %s Initail Trade Size %s", self.contract.symbol, self.tf, trade_size,t.quantity)

                    if trade_size is not None:
        
                        order_status,_ = self.client.place_order(self.contract, 'MARKET', trade_size, 'BUY')
        
                        logger.info("Market Order PLaced For Averaging for Symbol %s ON TIMEFRAME %s TRADE SIZE TO EXTEND %s Initail Trade Size %s", self.contract.symbol, self.tf, trade_size,t.quantity)

        
                        if order_status is not None:
        
                            while True:
        
                                order_status = self._check_order_status(order_status.order_id)
        
                                if order_status.status == "filled":
                                    # Calculate the new take profit point
                                    desired_roi = self.take_profit / 100
        
                                    logger.info("Initial takeprofit at %s", t.start_price + (t.start_price * desired_roi))

                                    total_position_size = t.quantity + trade_size
        
                                    weighted_avg_price = ((t.start_price * t.quantity) + (order_status.avg_price * trade_size)) / total_position_size


                                    logger.info("Weighted Average Price %s", weighted_avg_price)

                                    target_value = weighted_avg_price + (weighted_avg_price * self.take_profit / 100)
        
                                    logger.info("Target Value %s", target_value)

                                    new_takeprofit = target_value
        
                                    logger.info("New takeprofit Will be set at %s", new_takeprofit)

                                    logger.info("Modifying TP MARKET ORDER For Symbol %s ON TIMEFRAME %s TRADE SIZE TO EXTEND %s", self.contract.symbol, self.tf, trade_size)

                                    new_tp_order,_ =  self.client.place_order( self.contract, 'TAKE_PROFIT_MARKET', t.quantity, 'buy', absolute_tp=new_takeprofit)

                                    t.quantity += trade_size
                                    


        
                                    t.avg_price = weighted_avg_price


                                    if new_tp_order is None:
        
                                        logger.info("COULDNOT UPDATE TAKEPROFIT SUCCESSFULLY WHILE AVERAGING SYMBOL %s ",self.contract.symbol)
                                        
                                        
                                        sendEmail(f"ERROR COULD NOT AVERAGE TRADE PROPERLY {t.id}\nInitial Takeprofit {t.start_price + (t.start_price * desired_roi)}\n New take profit {new_takeprofit}\n Trade quantity {t.quantity}\n Average Price {t.avg_price}\n Total Invested {(t.avg_price * t.quantity)/self.leverage}",subject="AVERAGING TRADE")    
                                        
                                        break


                                    t.averaged =True
                                    
                                    t.tp = float(new_takeprofit)


                                    sendEmail(f"AVERAGING TRADE {t.id}\nInitial Takeprofit {t.start_price + (t.start_price * desired_roi)}\n New take profit {t.tp}\n Trade quantity {t.quantity}\n Average Price {t.avg_price}\n Total Invested {(t.avg_price * t.quantity)/self.leverage}",subject="AVERAGING TRADE")
                                    
                                    logger.info(f"AVERAGING TRADE {t.id}\nInitial Takeprofit {t.start_price + (t.start_price * desired_roi)}\n New take profit {t.tp}\n Trade quantity {t.quantity}\n Average Price {t.avg_price}\n Total Invested {(t.avg_price * t.quantity)/self.leverage}")
                                    
                                    break

                        else:
                            logger.info('Order Status Was None For Averaging Market Order')
                    
            if t.side == 'short':
                # print(f"{self.contract.candles[self.tf][-1].close} >= { entry + (entry*0.01)}")
                if self.contract.candles[self.tf][-1].close >= entry + (entry*0.01):
                    
                    trade_size = self.client.get_trade_size(self.contract, self.contract.candles[self.tf][-1].close, self.trade_amount, leverage=self.leverage)
                    
                    logger.info("AVERAGING REQUIREMENT FULFILED FOR SYMBOL %s ON TIMEFRAME %s TRADE SIZE TO EXTEND %s Initail Trade Size %s", self.contract.symbol, self.tf, trade_size,t.quantity)

                    
                    if trade_size is not None:
                    
                        order_status,_ = self.client.place_order(self.contract, 'MARKET', trade_size, 'SELL')
                    
                        logger.info("Market Order PLaced For Averaging for Symbol %s ON TIMEFRAME %s TRADE SIZE TO EXTEND %s Initail Trade Size %s", self.contract.symbol, self.tf, trade_size,t.quantity)

                        if order_status is not None:
                    
                            while True:
                    
                                order_status = self._check_order_status(order_status.order_id)
                    
                                if order_status.status == "filled":
                                    # Calculate the new take profit point
                    
                                    desired_roi = self.take_profit / 100
                    
                                    logger.info("Initial takeprofit at %s", t.start_price - (t.start_price * desired_roi))

                                    total_position_size = t.quantity + trade_size
                    
                                    weighted_avg_price = ((t.start_price * t.quantity) + (order_status.avg_price * trade_size)) / total_position_size
                    
                                    logger.info("Weighted Average Price %s", weighted_avg_price)

                                    target_value = weighted_avg_price - (weighted_avg_price * self.take_profit / 100)
                    
                                    logger.info("Target Value %s", target_value)

                                    new_takeprofit = target_value
                                    
                                    logger.info("New takeprofit Will be set at %s", new_takeprofit)

                                    logger.info("Modifying TP MARKET ORDER For Symbol %s ON TIMEFRAME %s TRADE SIZE TO EXTEND %s", self.contract.symbol, self.tf, trade_size)
                                    
                                    new_tp_order,_ = self.client.place_order(self.contract, 'TAKE_PROFIT_MARKET', t.quantity, 'sell', absolute_tp=new_takeprofit)
                                    
                                    
                                    t.quantity += trade_size
                                    
                                    
                                    t.avg_price = weighted_avg_price
                                    
                                    
                                    if new_tp_order is None:
                                        logger.info("COULDNOT UPDATE TAKEPROFIT SUCCESSFULLY WHILE AVERAGING SYMBOL %s ",self.contract.symbol)
                                    
                                        sendEmail(f"ERROR COULD NOT AVERAGE TRADE PROPERLY {t.id}\nInitial Takeprofit {t.entry_price + (t.entry_price * desired_roi)}\n New take profit {new_takeprofit}\n Trade quantity {t.quantity}\n Average Price {t.start_price}\n Total Invested {(t.start_price * t.quantity)/self.leverage}",subject="AVERAGING TRADE")                                   
                                       
                                        logger.info(f"ERROR COULD NOT AVERAGE TRADE PROPERLY {t.id}\nInitial Takeprofit {t.entry_price + (t.entry_price * desired_roi)}\n New take profit {new_takeprofit}\n Trade quantity {t.quantity}\n Average Price {t.start_price}\n Total Invested {(t.start_price * t.quantity)/self.leverage}")
                                        break
                                    
                                    t.averaged =True                                    
                                    t.tp = float(new_takeprofit)
                                    
                                    sendEmail(f"AVERAGING TRADE {t.id}\nInitial Takeprofit {t.start_price - (t.start_price * desired_roi)}\n New take profit {t.tp}\n Trade quantity {t.quantity}\n Average Price {t.avg_price}\n Total Invested {(t.avg_price * t.quantity)/self.leverage}",subject="AVERAGING TRADE")                                   
                                    break

                        else:
                            logger.info('Order Status Was None For Averaging Market Order')
        
        
        
        self.averagingMutex.release()    


    def _update_leverage(self)->int:

        leverage = self.leverage
        # stoploss 6
        if self.stop_loss == 6 :
            leverage = 4
        # stoploss 7-9
        elif self.stop_loss > 6 and self.stop_loss < 10:
            leverage = 3

        # stoploss 10-14
        elif self.stop_loss > 9 and self.stop_loss < 15:
            leverage = 2

        elif self.stop_loss > 14:
            leverage = 1
        self.client.change_leverage(leverage,self.contract)

        return leverage
        
    def _open_position(self, signal_result: int,stoploss=None):


#         if self.stat_name == 'TEST':
#                 openPos = self.contract.candles[self.tf][-1].close
#                 order_side = "buy" if signal_result == 1 else "sell"

#                 tp = openPos + (openPos * self.take_profit/100) if order_side == 'buy' else openPos - (openPos * self.take_profit/100 )
#                 sl = openPos + (openPos * self.stop_loss/100 ) if order_side == 'sell' else openPos - (openPos * self.stop_loss/100)


#                 signal = Signal(self.tf,self.contract,self.stat_name,order_side,openPos,sl,tp)
#                 msg = f"""
# Signal Id : {signal.id}
# Open Position : {openPos}
# TP: {tp} ({self.take_profit} %)
# SL : {sl} ({self.stop_loss} %)
#                     """
#                 self.send_msg_telegram(msg,signal.id)
#                 self.signals[signal.id] = signal
#                 return



        if self.tf in ['15m','30m','1h'] and 'Long Wick' in self.stat_name:
            return 
 
        if self.stat_name == 'Tweezer Bottom' and self.tf in ['15m','30m','1h','4h']:
            return
 
        self.tradeMutex.acquire()
        
        
        if stoploss is not None:
            self.stop_loss = (abs(self.contract.candles[self.tf][-1].close - stoploss)/self.contract.candles[self.tf][-1].close)*100 
        
        trades_open = [t for t in self.client.active_trades if t.status =='open'] 
        logger.info("YOU HAVE CURRENTLY %s Positions OPEN AND ARE TRYING TO OPEN A NEW ONE",len(trades_open))
        if not (len(trades_open) < self.client.trades_allowed):
            logger.info("Max Trades Limit Reached cannot Open New Position")
            sendEmail("Max Trades Limit Reached cannot Open New Position")
            self.tradeMutex.release()
            return
        
        ## Check if there is an existing open trade for the same coin
        for t in trades_open:
            if t.contract.symbol == self.contract.symbol:
                logger.info(f"Trade For symbol {self.contract.symbol} Already Open Cannot Open New Position")
                sendEmail(f"Trade For symbol {self.contract.symbol} Already Open Cannot Open New Position \n{t.id}")
                self.tradeMutex.release()
                return



        self.leverage =  self._update_leverage()
        

        
        
        trade_size = self.client.get_trade_size(self.contract, self.contract.candles[self.tf][-1].close, self.trade_amount,leverage = self.leverage)


        
        if trade_size is None:
            return

        order_side = "buy" if signal_result == 1 else "sell"
        position_side = "long" if signal_result == 1 else "short"

        logger.info(f"{position_side.capitalize()} signal on {self.contract.symbol} {self.tf}")


            ## STOP_MARKET/TAKE_PROFIT_MARKET	stopPrice First Send Stop_Market Request then send Take_profit_market Request
        startPos = self.contract.candles[self.tf][-1].close


        # MARKET ORDER
        order_status,_ = self.client.place_order(self.contract, "MARKET", trade_size, order_side )
        print(order_status)
        # TAKE_PROFIT_MARKET

        # modify place order to return tp price and sl price and save those when creating a trade 
        order_status_tp, order_tp = self.client.place_order(self.contract, "TAKE_PROFIT_MARKET", trade_size, order_side,last_price = startPos,tp =self.take_profit )
        if order_status_tp is not None:
            logger.info("TAKE PROFIT ORDER SUCCESSFULLY PLACE")
        else:
            sendEmail(f"Market Order placed but encountered some error while placing takeprofit order please manually place takeprofit at TP: {startPos + (startPos*self.take_profit/100) if position_side == 'long' else startPos - (startPos*self.take_profit/100)}({self.take_profit} %)")
        
        # STOP_MARKET
        # modify place order to return tp price and sl price and save those when creating a trade 
        order_status_sl,order_sl = self.client.place_order(self.contract, "STOP_MARKET", trade_size, order_side,last_price = startPos,sl = self.stop_loss )
        if order_status_sl is not None:
            logger.info("STOPLOSS ORDER SUCCESSFULLY PLACE")

        else:
            sendEmail(f"Market Order placed but encountered some error while placing Stoploss order please manually place Stoploss order at SL: {startPos - (startPos*self.take_profit/100) if position_side == 'long' else startPos + (startPos*self.take_profit/100)}  ({self.take_profit} %)")

        if order_status is not None:

            while True:
                order_status = self._check_order_status(order_status.order_id)

                avg_fill_price = order_status.avg_price

                if order_status.status == "filled":
                    logger.info(f'Strart price = {startPos} ,  Avg fill price  = {avg_fill_price} strategy {self.stat_name} tf  = {self.tf} contract = {self.contract.symbol}')
                    new_trade = Trade({"time": int(time.time() * 1000),'timeframe':self.tf, "entry_price": startPos,
                                        "contract": self.contract, "strategy": self.stat_name, "side": position_side,
                                        "status": "open", "pnl": 0, "quantity": trade_size, "entry_id": order_status.order_id,
                                        "sl_order_id":order_status_sl.order_id,"tp_order_id":order_status_tp.order_id,
                                        'avg_price':avg_fill_price
                                        })
                    new_trade.tp = order_tp
                    new_trade.sl = order_sl
                    logger.info(f"Trade Stoploss and take profits are {new_trade.tp} , {new_trade.sl} ")
                    self.trades.append(new_trade)
                    self.client.active_trades.append(new_trade)
                   
                    msg = f"""
Order Placed
T Id : {new_trade.id}
Open Position : {order_status.avg_price}
TP: {startPos + (startPos*self.take_profit/100) if position_side == 'long' else startPos - (startPos*self.take_profit/100)}({self.take_profit} %)
SL : {startPos - (startPos*self.take_profit/100) if position_side == 'long' else startPos + (startPos*self.take_profit/100)}  ({self.stop_loss} %)
Amount : $ {self.trade_amount}
Leverage : {self.leverage}X
                    """

                    # logger.info(f"{msg}")

                    try:
                        self._write_history(tradeTaken=True,update=True )
                    except:
                        pass
                    self.send_msg(msg,add_info=False,trade=True)
                   
                    

                    break
       
       
        self.tradeMutex.release()

    def _write_history(self,tradeTaken=False,update=False):
        if self.stat_name =="TEST":
            return
        history = {
                  'symbol':self.contract.symbol,
                  'strategy':self.stat_name,
                  'orderSide':self._signal_type,
                  'timeframe':self.tf,
                  'timestamp':self.last_candle.timestamp,
                  'takeprofit_pct':self.take_profit,                                
                  'stoploss_pct':self.stop_loss,
                  'trade_taken':tradeTaken,
                  'rsi' : self.last_candle.rsi
          }
        try:
            if update:
                data = []
                with open(self.history_csv, 'r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        data.append(row)
            
                data[-1] = history

                with open(self.history_csv, 'w', newline='') as csvfile:
                    fieldnames = data[0].keys()
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for row in data:
                        writer.writerow(row)
                logger.info("TRADE GENERATED AND TAKEN CSV UPDATED SUCCESSFULLY")
            else:
                with open(self.history_csv, 'a') as f:
                    for key, val in history.items():
                        f.write(f"{val},")
                    f.write('\n')
                logger.info("TRADE GENERATED For Symbol %s Tf %s Using %s WRITTEN TO CSV SUCCESSFULLY",self.contract.symbol,self.tf,self.stat_name)

        except Exception as e:
            logger.info("error Writting to file",e)
    
    def _check_tp_sl(self,data = None):


        if not self.sl_tp_mutex.acquire(False):
            # logger.info(f"MUTEX COULD NOT BE AQUIRED FOR CHEKING TRADES {self.stat_name} {self.tf} {self.contract.symbol}")
            return
        try:
            for trade in self.trades:

                # logger.info(f"CHECKING TRADE ID: {trade.id} , STATUS: {trade.status} , SIDE : {trade.side} , avg_price : {trade.avg_price} start_price :{trade.start_price} ")

                if trade.status == "open":     
                    tp_triggered = False
                    sl_triggered = False
                    price = self.contract.candles[self.tf][-1].close
                    high = self.contract.candles[self.tf][-1].high
                    low  = self.contract.candles[self.tf][-1].low
                    tp_point = 0
                    sl_point = 0


                    if trade.side == "long":
                        tp_point = trade.tp
                        sl_point = trade.sl
                        trade.pnl = (float(data['k']['c']) - trade.avg_price) * (trade.quantity ) 
                        

                        # logger.info(f"CHECKING TRADE ID: {trade.id} , STATUS: {trade.status} , SIDE : {trade.side} , avg_price : {trade.avg_price} start_price :{trade.start_price} tp : {tp_point} sl :{sl_point} current price : {price} ,PNL : {trade.pnl}")
                        
                        if low <= sl_point:
                                sl_triggered = True
                                
                                trade.pnl = (float(low) - trade.avg_price) * (trade.quantity ) 
                        
                        if high >= tp_point:    
                            tp_triggered = True
                            trade.pnl = (float(high) - trade.avg_price) * (trade.quantity ) 
                    
                    
                    elif trade.side == "short":
                        trade.pnl = (trade.avg_price - float(data['k']['c']) ) * (trade.quantity)
                        tp_point = trade.tp
                        sl_point = trade.sl


                        # logger.info(f"CHECKING TRADE ID: {trade.id} , STATUS: {trade.status} , SIDE : {trade.side} , avg_price : {trade.avg_price} start_price :{trade.start_price} tp : {tp_point} sl :{sl_point} current price : {price} ,PNL : {trade.pnl}")
                        
                        
                        
                        
                        if high >= sl_point:
                            sl_triggered = True
                            trade.pnl = ( trade.avg_price - float(high)) * (trade.quantity ) 
                    
                        elif low <= tp_point:
                            tp_triggered = True
                            trade.pnl = ( trade.avg_price - float(low)) * (trade.quantity ) 

                    if tp_triggered or sl_triggered:
                            # CONFIRM IF POSITION IS CLOSED OR NOT
                            result = "TP hit" if tp_triggered else "SL hit"
                            logger.info(f"{'Stop loss' if sl_triggered else 'Take profit'} for {self.contract.symbol} {self.tf} Trade entry {trade.avg_price} Trade Quantity : {trade.quantity} , PNL {trade.pnl}")
                            msg = f"\nOrder Closed\n{result} \nT Id = {trade.id}\n Close position : {price}\n PNL : {trade.pnl}\n Binance Fees : {(trade.quantity * trade.avg_price) * 0.000500 } \n\nOpen Position : {trade.start_price}\nSide : {self._signal_type}"
                            
                            if trade.averaged:
                                msg+=f"\nTP: {trade.tp} ({self.take_profit}%)\nSL : {sl_point} ({self.stop_loss}%)\nInvestment Amount : $ {self.trade_amount * 2}\nLeverage : {self.client.leverage}X"

                            else:
                                msg+=f"\nTP: {tp_point} ({self.take_profit}%)\nSL : {sl_point} ({self.stop_loss}%)\nInvestment Amount : $ {self.trade_amount}\nLeverage : {self.client.leverage}X"

                            self.send_msg(msg,openPos=trade.entry_price,trade=True,add_info=False)

                            trade.status = "closed"
                            self.ongoing_position = False


                            position = self.client.get_open_position(self.contract.symbol)[0]
                            logger.info(f"position while confirming trade close {position['entryPrice'] if position is not None else "None"}")
                            if position is not None:
                                if position['entryPrice'] !=  '0.0' :
                                    sendEmail(f"RECHECK POSITION CLOSEING for Trade {trade.id}\n{result} According to BOT but position remains open")

                                    logger.error(f"Recheck Position Close ")
                                    
                                    return
                        



                    else:
                        position = self.client.get_open_position(self.contract.symbol)[0]
                        # logger.info(f"CHECKING POSITION {position}")
                        if position['entryPrice'] ==  '0.0' :
                            
                            logger.error(f"POSITION CLOSE Was NOT REGISTERED, Trade ID {trade.id} ")
                            msg = f"Order Closed Manually OR close not registered \nT Id = {trade.id}"
                            sendEmail(msg)

                            trade.status = "closed"
                            self.ongoing_position = False
                            
        finally: 
                self.sl_tp_mutex.release()
                self.averaging()
                return

   


 
    def close_position(self,trade:Trade):
            print("strategies close position called")
            order_side = "SELL" if trade.side == "long" else "BUY"
            order_status,_ = self.client.place_order(self.contract, "MARKET", trade.quantity, order_side)

            if order_status is not None:
                trade.status = "closed"
                self.ongoing_position = False
                return 'Trade closed successfully'
            else:
                print('could not close trade')

    
    
    # def open_position_telegram(self,signal_id,trade):
    #     try:
    #         if trade == 0:
    #             del self.signals[signal_id]
    #             return 
    #         self.tradeMutex.release()
            
    
    #         stoploss = self.signals[signal_id].sl


    #         takeprofit = self.signals[signal_id].tp
    #         side = self.signals[signal_id].side
    #         position_side = "long" if side == 'buy' else "short"

    #         # TODO
    #         # CHECK IF THERE IS STILL A DIFFERENCE BETWEEN STOPLOSS TAKEPROFIT AND CURRECT PRICE

    #         lastCandle = self.contract.candles[self.tf][-1]

    #         sl_triggered = False
    #         tp_triggered = True
    #         if position_side == 'long':
    #             if lastCandle.high >= takeprofit:
    #                 tp_triggered = True
    #                 sl_triggered = True
    #             elif lastCandle.low <= stoploss: 
    #                 sl_triggered = True
    #         else:
    #             if lastCandle.high >= stoploss:
    #                 sl_triggered = True
    #             elif lastCandle.low <= takeprofit: 
    #                 tp_triggered = True
            


    #         if tp_triggered or sl_triggered:
    #             msg = f"{'Stop loss' if sl_triggered else 'Take profit'} for {self.signals[signal_id].contract.symbol} {self.tf} entry {self.signals[signal_id].entry_price}  has already been triggered"
    #             logger.info(msg)
                
               
    #             self.send_msg(msg,trade=True,add_info=False)

    #             # send back message and that stoploss or takeprofit has already been triggered
    #             del self.signals[signal_id]

                
    #         trades_open = [t for t in self.client.active_trades if t.status =='open'] 
    #         logger.info("YOU HAVE CURRENTLY %s Positions OPEN AND ARE TRYING TO OPEN A NEW ONE",len(trades_open))
    #         if not (len(trades_open) < self.client.trades_allowed):
    #             logger.info("Max Trades Limit Reached cannot Open New Position")
    #             sendEmail("Max Trades Limit Reached cannot Open New Position")
    #             self.tradeMutex.release()
    #             return
            
    #         ## Check if there is an existing open trade for the same coin
    #         for t in trades_open:
    #             if t.contract.symbol == self.contract.symbol:
    #                 logger.info(f"Trade For symbol {self.contract.symbol} Already Open Cannot Open New Position")
    #                 sendEmail(f"Trade For symbol {self.contract.symbol} Already Open Cannot Open New Position \n{t.id}")
    #                 self.tradeMutex.release()
    #                 return


    #         self.leverage = self._update_leverage()
    #         trade_size = self.client.get_trade_size(self.contract, self.contract.candles[self.tf][-1].close, self.trade_amount,leverage = self.leverage)


            
    #         if trade_size is None:
    #             return

    #         order_side = side
            
    #         startPos = self.contract.candles[self.tf][-1].close
    #         stoploss_pct = (abs ( startPos -  stoploss ) / startPos )  * 100 
    #         takeprofit_pct = (abs ( startPos -  takeprofit ) / startPos )  * 100 
        
    #         # MARKET ORDER
    #         order_status,_ = self.client.place_order(self.contract, "MARKET", trade_size, order_side )
    #         print(order_status)
        
    #         # TAKE_PROFIT_MARKET
    #         order_status_tp = self.client.place_order(self.contract, "TAKE_PROFIT_MARKET", trade_size, order_side,last_price = startPos,tp =takeprofit_pct )
    #         if order_status_tp is not None:
    #             logger.info("TAKE PROFIT ORDER SUCCESSFULLY PLACE")
        
    #         # STOP_MARKET
    #         order_status_sl = self.client.place_order(self.contract, "STOP_MARKET", trade_size, order_side,last_price = startPos,sl = stoploss_pct )
    #         if order_status_sl is not None:
    #             logger.info("TAKE PROFIT ORDER SUCCESSFULLY PLACE")



    #         if order_status is not None:                
    #             while True:
    #                 order_status = self._check_order_status(order_status.order_id)
    #                 avg_fill_price = order_status.avg_price
    #                 if order_status.status == "filled":

    #                     new_trade = Trade({"time": int(time.time() * 1000),'timeframe':self.tf, "entry_price": startPos,
    #                                         "contract": self.contract, "strategy": self.stat_name, "side": position_side,
    #                                         "status": "open", "pnl": 0, "quantity": trade_size, "entry_id": order_status.order_id,
    #                                         "sl_order_id":order_status_sl.order_id,"tp_order_id":order_status_tp.order_id
    #                                         })
                        
    #                     self.trades.append(new_trade)
    #                     self.client.active_trades.append(new_trade)
                    
                    
    #                     msg = f"""
    # Order Placed Using Telegram Poll
    # T Id : {new_trade.id}
    # Open Position : {order_status.avg_price}
    # TP: { takeprofit }( { takeprofit_pct } %)
    # SL : { stoploss } ({ stoploss_pct } %)
    # Amount : $ {self.trade_amount}
    # Leverage : {self.leverage}X
    #                     """
    #                     self.send_msg(msg,add_info=False,trade=True)
    #                     break
        
    #         del self.signals[signal_id]
    #         self.tradeMutex.release()


    #     except KeyError as e:
    #         return "Signal Does not exist was not generated or has expired"


class TestStrategy(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "TEST","short",averaging_strat=True)
        for i in range(-5,0):
            print(f"TEST STRAT {i}   {self.contract.candles[self.tf][i].timestamp}")
        print(self.averaging_strat)
    
    
    def _check_signal(self):
        lastCandle =  self.contract.candles[self.tf][-2]
 
        logger.info("Opening short position using test Strategy")
        msg = f"Short Signal for {self.contract.symbol} on timeframe {self.tf} using {self.stat_name}"
        tradeId = f'{self.stat_name}_{self.contract.symbol}_{self.tf}'
        self.send_msg(msg)
        t = Thread(target=self._write_history).start()

        return -1
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                self._open_position(signal_result)

                pass

        return signal_result
    



# Done
## relaxing pattern a little
class HammerStrategy(Strategy):
  
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
    
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Hammer","long")
    
    def _check_signal(self) -> int:
 
        logger.info("CHECKING SIGNAL FOR %s %s  ",  self.contract.symbol, self.stat_name)

        if self.find_trend() == -1:
            logger.info("DOWNWARD TREND FOR %s %s  ",  self.contract.symbol, self.tf)

            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]


            # Requirements for a valid hammer
            # 1. It should be in a downtrend with trend definitions relaxed this should work
            # 2. First candle i.e starting candle should be bearish
            # 3. Second Candle i.e trading candidate should be hammer
            #       Requirements for a hammer     
            #           3a. Must be bullish
            #           3b. Long tail with small body tail must be atleast 1.5 times the body
            #           3c. close and high of hammer must be equal allow 1-2% nip

 
                         
                        
                        
            if startingCandle.open > startingCandle.close:
                # bearish candle in downward trend requirment 2 
                if tradingCandidate.open < tradingCandidate.close: 
                    # bullish candle requirement 3a
                    if abs(tradingCandidate.close - tradingCandidate.open) *3 < abs(tradingCandidate.low - tradingCandidate.open):
                        # long tail requirement 3b
                        if tradingCandidate.high < (tradingCandidate.close + (tradingCandidate.close*0.001)):
                            # hammer body requirement 3c
                            print(f'Found an hammer for {self.contract.symbol} on timeframe {self.tf}')                       
                            msg = f"Long trade Signal for {self.contract.symbol} on timeframe {self.tf} using {self.stat_name}"
                            logger.info("%s", msg)

                            self.send_msg(msg)
                            t = Thread(target=self._write_history).start()
                            return 1
            
        
        return 0

    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        #loop over previous candles check if conditions are being met give back data that will be save in an csv or excel file
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        try:
            for i in range(3, len(previousCandles)):
                startingCandle = previousCandles[i-3]
                middleCandle = previousCandles[i-2]
                endingCandle = previousCandles[i-1]
                tradingCandidate = previousCandles[i]
                if startingCandle.open > startingCandle.close:
                    if middleCandle.open > middleCandle.close and middleCandle.close < (startingCandle.close - (abs(startingCandle.high - startingCandle.low ) * 0.02) ) :
                        if endingCandle.open > endingCandle.close and endingCandle.close < (middleCandle.close - (abs(middleCandle.high - middleCandle.low ) * 0.02) ):
                            
                            if tradingCandidate.close == tradingCandidate.high and tradingCandidate.low != tradingCandidate.open:
                                if abs(tradingCandidate.open - tradingCandidate.close) < abs((tradingCandidate.high - tradingCandidate.low) * 0.25):
                                    print(f'hammerfound')
                                    # if hammer is found we set take profit and stop loss
                                    # trade will be entered at the close of tradingCandidate
                                    # long trade
                                    start_pos = tradingCandidate.close
                                    tp_test = start_pos + (start_pos * tp_pct)
                                    sl_test = start_pos - (start_pos * sl_pct)
                                    j = 1
                                    obj = {
                                        'start_pos': start_pos,
                                        'tp_test': tp_test,
                                        'sl_test': sl_test,
                                        'enterTimestamp' : tradingCandidate.timestamp,
                                        'symbol': self.contract.symbol,
                                        'tf': timeframe
                                    }
                                    while True:
                                        try:
                                            testCandle = previousCandles[i+j]
                                            obj['finalcandleTimestamp'] =  None
                                            obj['finalcandleClose'] = None
                                            obj['tp_hit'] = None
                                            obj['sl_hit'] = None
                                            obj['finalcandleHigh'] = None
                                            obj['finalcandleLow'] = None
                                            obj['finalcandleOpen'] = None
                                            if testCandle.high >= tp_test:
                                                # take profit hit at j next candle than candidate candle
                                                obj['exitTimestamp'] = testCandle.timestamp
                                                obj['candleForward'] = j
                                                obj['exit_pos'] = testCandle.high
                                                obj['tp_hit'] = True
                                                obj['sl_hit'] = False
                                                data.append(obj)
                                                break
                                            elif testCandle.low <= sl_test:
                                                # stop loss hit at j next candle than candidate candle
                                                obj['exitTimestamp'] = testCandle.timestamp
                                                obj['candleForward'] = j
                                                obj['exit_pos'] = testCandle.low
                                                obj['tp_hit'] = False
                                                obj['sl_hit'] = True
                                                data.append(obj)
                                                break
                                            j = j + 1
                                        except IndexError as e:
                                            print('End of candles but stoploss or tp not hit yet')
                                            testCandle = previousCandles[i]
                                            obj['finalcandleTimestamp'] =  testCandle.timestamp
                                            obj['finalcandleClose'] = testCandle.close
                                            obj['tp_hit'] = False
                                            obj['sl_hit'] = False
                                            obj['finalcandleHigh'] = testCandle.high
                                            obj['finalcandleLow'] = testCandle.low
                                            obj['finalcandleOpen'] = testCandle.open
                                            data.append(obj)
                                            break
                                    
                
        except Exception as e:
            print('Error in backtesting' , e)
        return data

    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result

## relaxing pattern a little
class InvertedHammerStrategy(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Inverted Hammer","long")

    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        
        previousCandles =  self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0

        try:
            for i in range(3, len(previousCandles)):
                startingCandle = previousCandles[i-3]
                middleCandle = previousCandles[i-2]
                endingCandle = previousCandles[i-1]
                tradingCandidate = previousCandles[i]
                if startingCandle.open < startingCandle.close:
                    if middleCandle.open < middleCandle.close and middleCandle.close > (startingCandle.close + (abs(startingCandle.high - startingCandle.low ) * 0.02) ) :
                        if endingCandle.open < endingCandle.close and endingCandle.close > (middleCandle.close + (abs(middleCandle.high - middleCandle.low ) * 0.02) ):
                            
                            if tradingCandidate.close == tradingCandidate.low and tradingCandidate.high != tradingCandidate.open:
                                if abs(tradingCandidate.open - tradingCandidate.close) < abs((tradingCandidate.high - tradingCandidate.low) * 0.25):
                                    print(f'hammerfound' ,timeframe)
                                    start_pos = tradingCandidate.close
                                    tp_test = start_pos - (start_pos * tp_pct)
                                    sl_test = start_pos + (start_pos * sl_pct)
                                    j = 1
                                    obj = {
                                        'start_pos': start_pos,
                                        'tp_test': tp_test,
                                        'sl_test': sl_test,
                                        'enterTimestamp' : tradingCandidate.timestamp,
                                        'symbol': self.contract.symbol,
                                        'tf': timeframe
                                    }
                                    print('start_pos',start_pos)
                                    print('tp_test',tp_test)
                                    print('sl_test',sl_test)
                                    print('enterTimestamp',tradingCandidate.timestamp)
                                    while True:
                                        try:
                                            testCandle = previousCandles[i+j]
                                            obj['finalcandleTimestamp'] =  None
                                            obj['finalcandleClose'] = None
                                            obj['tp_hit'] = None
                                            obj['sl_hit'] = None
                                            obj['finalcandleHigh'] = None
                                            obj['finalcandleLow'] = None
                                            obj['finalcandleOpen'] = None
                                            if testCandle.low <= tp_test:
                                                # take profit hit at j next candle than candidate candle
                                                obj['exitTimestamp'] = testCandle.timestamp
                                                obj['candleForward'] = j
                                                obj['exit_pos'] = testCandle.low
                                                obj['tp_hit'] = True
                                                obj['sl_hit'] = False
                                                data.append(obj)
                                                break
                                            elif testCandle.high >= sl_test:
                                                obj['exitTimestamp'] = testCandle.timestamp
                                                obj['candleForward'] = j
                                                obj['exit_pos'] = testCandle.high
                                                obj['tp_hit'] = False
                                                obj['sl_hit'] = True
                                                data.append(obj)
                                                break
                                            j = j + 1
                                        except IndexError as e:
                                            print('End of candles but stoploss or tp not hit yet')
                                            testCandle = previousCandles[i]
                                            obj['finalcandleTimestamp'] =  testCandle.timestamp
                                            obj['finalcandleClose'] = testCandle.close
                                            obj['tp_hit'] = False
                                            obj['sl_hit'] = False
                                            obj['finalcandleHigh'] = testCandle.high
                                            obj['finalcandleLow'] = testCandle.low
                                            obj['finalcandleOpen'] = testCandle.open
                                            data.append(obj)
                                            break
                                    
                
        except Exception as e:
            print('Error in backtesting' , e)
        return data

    def _check_signal(self) -> int:
 

        if self.find_trend() == -1:
            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]


            # Requirements for a valid inverted hammer
            # 1. It should be in an upward trend
            # 2. First candle i.e starting candle should be bullish
            # 3. Second Candle i.e trading candidate should be an inverted hammer
            #       Requirements for a hammer     
            #           3a. Must be bearish
            #           3b. Long tail with small body tail must be atleast 1.5 times the body
            #           3c. close and low of inverted hammer must be equal allow a little nip



            if startingCandle.open < startingCandle.close:
                # bullish candle in downward trend requirment 2 
                if tradingCandidate.open > tradingCandidate.close: 
                    # bearish candle requirement 3a
                    if abs(tradingCandidate.open - tradingCandidate.close) *3 < abs(tradingCandidate.high - tradingCandidate.open):
                        # long tail requirement 3b
                        if tradingCandidate.low > (tradingCandidate.close - (tradingCandidate.close*0.001)):
                    # hammer body requirement 3c
                            print(f'Found an inverted hammer for {self.contract.symbol} on timeframe {self.tf}')                        
                            msg = f"Short trade Signal for {self.contract.symbol} on timeframe {self.tf} using {self.stat_name}"
                            logger.info("%s", msg)
                            self.send_msg(msg)
                            t = Thread(target=self._write_history).start()

                            return 1
            
        
        return 0


    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result


class TweezerTopStrategy(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):


        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            
            '15m':{
                'sl':2,
                'tp':1
            },
            '30m':{
                'sl':2,
                'tp':1
            },     
            '1h':{
                'sl':2,
                'tp':1
            },
            '4h':{
                'sl':5,
                'tp':1
            },
            '1d':{
                'sl':3,
                'tp':2
            },
            '3d':{
                'sl':2,
                'tp':1
            },
            '1w':{
                'sl':15,
                'tp':4
            },
            '1M':{
                'sl':3,
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount, sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Tweezer Top","short")
    
    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                if startingCandle.close > startingCandle.open and tradingCandidate.close < tradingCandidate.open and tradingCandidate.close <= startingCandle.open and tradingCandidate.low < startingCandle.low:
                    print(f'Trade found {timeframe} , {self.contract.symbol}')
                    start_pos = tradingCandidate.close
                    tp_test = start_pos - (start_pos * tp_pct)
                    sl_test = start_pos + (start_pos * sl_pct)
                    j = 1
                    obj = {
                        'start_pos': start_pos,
                        'tp_test': tp_test,
                        'sl_test': sl_test,
                        'enterTimestamp' : tradingCandidate.timestamp,
                        'symbol': self.contract.symbol,
                        'tf': timeframe
                    }
                    print('start_pos',start_pos)
                    print('tp_test',tp_test)
                    print('sl_test',sl_test)
                    print('enterTimestamp',tradingCandidate.timestamp)
                    while True:
                        try:
                            testCandle = previousCandles[i+j]
                            obj['finalcandleTimestamp'] =  None
                            obj['finalcandleClose'] = None
                            obj['tp_hit'] = None
                            obj['sl_hit'] = None
                            obj['finalcandleHigh'] = None
                            obj['finalcandleLow'] = None
                            obj['finalcandleOpen'] = None
                            if testCandle.low <= tp_test:
                                obj['exitTimestamp'] = testCandle.timestamp
                                obj['candleForward'] = j
                                obj['exit_pos'] = testCandle.low
                                obj['tp_hit'] = True
                                obj['sl_hit'] = False
                                data.append(obj)
                                break
                            elif testCandle.high >= sl_test:
                                obj['exitTimestamp'] = testCandle.timestamp
                                obj['candleForward'] = j
                                obj['exit_pos'] = testCandle.high
                                obj['tp_hit'] = False
                                obj['sl_hit'] = True
                                data.append(obj)
                                break
                            j = j + 1
                        except IndexError as e:
                            print('End of candles but stoploss or tp not hit yet')
                            testCandle = previousCandles[i]
                            obj['finalcandleTimestamp'] =  testCandle.timestamp
                            obj['finalcandleClose'] = testCandle.close
                            obj['tp_hit'] = False
                            obj['sl_hit'] = False
                            obj['finalcandleHigh'] = testCandle.high
                            obj['finalcandleLow'] = testCandle.low
                            obj['finalcandleOpen'] = testCandle.open
                            data.append(obj)
                            break
                                    
                
        return data



    def _check_signal(self) -> int:

        logger.info("CHECKING SIGNAL FOR %s %s  ",  self.contract.symbol, self.stat_name)

        t =self.find_trend() 
        if t== 0:
            return 0
        elif t ==1:
            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]
            print(f'Upward trend for {self.contract.symbol} on timeframe {self.tf} using {self.stat_name}')
            # add requirement for body of trading and starting candle body
            if startingCandle.close > startingCandle.open and tradingCandidate.close < tradingCandidate.open and tradingCandidate.close <= startingCandle.open and tradingCandidate.low < startingCandle.low and abs(startingCandle.close - startingCandle.open) > abs(startingCandle.low - startingCandle.high) /8 and abs(tradingCandidate.close - tradingCandidate.open) > abs(tradingCandidate.low - tradingCandidate.high) /8:
                print(f'Tweezer Found indicating Short trade {self.contract.symbol} on timeframe {self.tf} ')                            
                msg = f"Short trade Signal for  {self.contract.symbol} using tweezer Strategy on timeframe {self.tf}"
                logger.info("%s", msg)
                self.send_msg(msg)
                print("message sent")


                t = Thread(target=self._write_history).start()

                return -1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)

                pass

        return signal_result


class TweezerBottonStrategy(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float,):
        
        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            '15m':{
                'sl':1,
                'tp':1
            },
            '30m':{
                'sl':1,
                'tp':1
            },     
            '1h':{
                'sl':1,
                'tp':1
            },
            '4h':{
                'sl':1,
                'tp':1
            },
            '1d':{
                'sl':5,
                'tp':2
            },
            '3d':{
                'sl':5, 
                'tp':2
            },
            '1w':{
                'sl':5, 
                'tp':1
            },
            '1M':{
                'sl':5,
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount,  sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Tweezer Bottom","long")


    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles =  self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == -1:
                if startingCandle.close < startingCandle.open and tradingCandidate.close > tradingCandidate.open and tradingCandidate.close >= startingCandle.open and tradingCandidate.high > startingCandle.high:
                    print(f'Trade found {timeframe} , {self.contract.symbol}')
                    start_pos = tradingCandidate.close
                    tp_test = start_pos + (start_pos * tp_pct)
                    sl_test = start_pos - (start_pos * sl_pct)
                    j = 1
                    obj = {
                        'start_pos': start_pos,
                        'tp_test': tp_test,
                        'sl_test': sl_test,
                        'enterTimestamp' : tradingCandidate.timestamp,
                        'symbol': self.contract.symbol,
                        'tf': timeframe
                    }
                    print('start_pos',start_pos)
                    print('tp_test',tp_test)
                    print('sl_test',sl_test)
                    print('enterTimestamp',tradingCandidate.timestamp)
                    while True:
                        try:
                            testCandle = previousCandles[i+j]
                            obj['finalcandleTimestamp'] =  None
                            obj['finalcandleClose'] = None
                            obj['tp_hit'] = None
                            obj['sl_hit'] = None
                            obj['finalcandleHigh'] = None
                            obj['finalcandleLow'] = None
                            obj['finalcandleOpen'] = None
                            if testCandle.high >= tp_test:
                                obj['exitTimestamp'] = testCandle.timestamp
                                obj['candleForward'] = j
                                obj['exit_pos'] = testCandle.low
                                obj['tp_hit'] = True
                                obj['sl_hit'] = False
                                data.append(obj)
                                break
                            elif testCandle.low <= sl_test:
                                obj['exitTimestamp'] = testCandle.timestamp
                                obj['candleForward'] = j
                                obj['exit_pos'] = testCandle.high
                                obj['tp_hit'] = False
                                obj['sl_hit'] = True
                                data.append(obj)
                                break
                            j = j + 1
                        except IndexError as e:
                            print('End of candles but stoploss or tp not hit yet')
                            testCandle = previousCandles[i]
                            obj['finalcandleTimestamp'] =  testCandle.timestamp
                            obj['finalcandleClose'] = testCandle.close
                            obj['tp_hit'] = False
                            obj['sl_hit'] = False
                            obj['finalcandleHigh'] = testCandle.high
                            obj['finalcandleLow'] = testCandle.low
                            obj['finalcandleOpen'] = testCandle.open
                            data.append(obj)
                            break
                                    
                
        return data






    def _check_signal(self) -> int:

 

        t =self.find_trend() 
        if t== 0:
            return 0
        elif t == -1:
            
            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]
            # logger.info(f"STARTING CANDLE\nOPEN {startingCandle.open} ")
            if startingCandle.close < startingCandle.open and tradingCandidate.close > tradingCandidate.open and tradingCandidate.close >= startingCandle.open and tradingCandidate.high > startingCandle.high and abs(startingCandle.close - startingCandle.open) > abs(startingCandle.low - startingCandle.high) /8 and abs(tradingCandidate.close - tradingCandidate.open) > abs(tradingCandidate.low - tradingCandidate.high) /8 :

                msg = f"Long trade Signal for  {self.contract.symbol} using tweezer Strategy on timeframe {self.tf}"
                logger.info("%s", msg)
                self.send_msg(msg)
                print("message sent")
                t = Thread(target=self._write_history).start()
                return 1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle":
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                self._open_position(signal_result)
                pass

        return signal_result




## relaxing pattern a little
class BullishHaramiCross(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        """The best time frame for trading the Bearish Harami Cross candlestick pattern will depend on the trader's trading style and risk tolerance. Some traders may prefer to use it on shorter time frames such as the 1-hour or 4-hour chart, while others may use it on daily or weekly charts. However, a rule of thumb is the higher the timeframe, the stronger the pattern."""


        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            '15m':{
                'sl':3,
                'tp':1
            },
            '30m':{
                'sl':2,
                'tp':1
            },     
            '1h':{
                'sl':7,
                'tp':2
            },
            '4h':{
                'sl':18, #20
                'tp':9
            },
            '1d':{
                'sl':16,
                'tp':7
            },
            '3d':{
                'sl':18, #20
                'tp':9
            },
            '1w':{
                'sl':18, #20
                'tp':9
            },
            '1M':{
                'sl':3,
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount,  sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Bullish Harami Cross",'long',averaging_strat = True)
    
    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles  = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            motherCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == -1:
                if motherCandle.close < motherCandle.open:
                # very little cariation in open and close of the second candle
                    if (abs(tradingCandidate.close - tradingCandidate.open) < tradingCandidate.close * 0.02  ) and tradingCandidate.close > motherCandle.close and tradingCandidate.close < motherCandle.open and tradingCandidate.high !=tradingCandidate.close and  tradingCandidate.low != tradingCandidate.close and tradingCandidate.low !=tradingCandidate.close: 
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos + (start_pos * tp_pct)
                        sl_test = start_pos - (start_pos * sl_pct)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.high >= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.low <= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
                                        
                    
        return data




    def _check_signal(self) -> int:
        t =self.find_trend() 
 

        # Requirements for a bullsih HM 
        # 1. downward trend
        # 2. Bearish mother candle
        # 3. Trading candidate fully enclosed by mother candle
        # 4. Trading Candidate must be doji
        #       Requirements for doji candle 
        #           4a. open and close are very close together lets say about 0.04% 
        #           
        if t == 0:
            return 0
        elif t ==-1:
            # Requirement 1 
            motherCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2] 

            if motherCandle.close < motherCandle.open:
                # Requirement 2
                if motherCandle.low< tradingCandidate.low and motherCandle.high>tradingCandidate.high:
                    #requirement 3
                    if abs((tradingCandidate.close - tradingCandidate.open) / tradingCandidate.open) *100 <= 0.04:

                        print(f'Harami Cross found in a downward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                        msg = f"Long trade Signal for  {self.contract.symbol} using Harami Cross Strategy on timeframe {self.tf}"
                        logger.info("%s", msg)
                        self.send_msg(msg)
                        t = Thread(target=self._write_history).start()
                        return 1
        
       
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                self._open_position(signal_result)
                pass

        return signal_result
## relaxing pattern a little
class BearishHaramiCross(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
      
        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            '15m':{
                'sl':3,
                'tp':1
            },
            '30m':{
                'sl':3,
                'tp':1
            },     
            '1h':{
                'sl':3,
                'tp':1
            },
            '4h':{
                'sl':8,
                'tp':2
            },
            '1d':{
                'sl':3,
                'tp':1
            },
            '3d':{
                'sl':5,
                'tp':1
            },
            '1w':{
                'sl':3,
                'tp':1
            },
            '1M':{
                'sl':3,
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount, sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Bearish Harami Cross",'short',averaging_strat = True)
    
    def _check_signal(self) -> int:
        t =self.find_trend() 
        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        # Requirements for a bullsih HM 
        # 1. Upward trend
        # 2. Bullish mother candle
        # 3. Trading candidate fully enclosed by mother candle
        # 4. Trading Candidate must be doji
        #       Requirements for doji candle 
        #           4a. open and close are very close together lets say about 1-2% 
        #           
        if t == 0:
            return 0
        elif t ==1:
            # Requirement 1 
            motherCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2] 

            if motherCandle.close > motherCandle.open:
                # Requirement 2
                if motherCandle.low < tradingCandidate.low and motherCandle.high > tradingCandidate.high:
                    #requirement 3
                    if abs((tradingCandidate.close - tradingCandidate.open) / tradingCandidate.open) *100 <= 0.04:
                        # requirement 4a
                        print(f'Harami Cross found in a upward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                        msg = f"Short trade Signal for  {self.contract.symbol} using Harami Cross Strategy on timeframe {self.tf} "
                        logger.info("%s", msg)
                        self.send_msg(msg)
                        t = Thread(target=self._write_history).start()
                        return -1       
        return 0
    
    
    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):

            motherCandle  = previousCandles[i-1]
            tradingCandidate = previousCandles[i]


            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                if motherCandle.close > motherCandle.open:
                    # very little variation in open and close of the second candle
                    if (abs(tradingCandidate.close - tradingCandidate.open) < tradingCandidate.close * 0.02  ) and tradingCandidate.close < motherCandle.close and tradingCandidate.close > motherCandle.open and tradingCandidate.high != tradingCandidate.close and tradingCandidate.high !=tradingCandidate.close and  tradingCandidate.low != tradingCandidate.close and tradingCandidate.low !=tradingCandidate.close:
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos - (start_pos * tp_pct)
                            sl_test = start_pos + (start_pos * sl_pct)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.low <= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.high >= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        print('printing data' , data)      
        return data

    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                self._open_position(signal_result)
                pass

        return signal_result

## relaxing pattern a little
class DarkCloud(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
  
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Dark Cloud","short")
    


    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                if startingCandle.close >startingCandle.open and tradingCandidate.close< tradingCandidate.open and startingCandle.close != startingCandle.open:
                    if tradingCandidate.open > startingCandle.close and tradingCandidate.close < (abs(startingCandle.open - startingCandle.close)/2):
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos - (start_pos * 0.01)
                        sl_test = start_pos + (start_pos * 0.01)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.low <= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.high >= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
       
        print('printing data' , data)      
        return data


    def _check_signal(self) -> int:
        t = self.find_trend()
        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        if t == 0:
            return 0
            # Requirements for a dark cloud cover 
            # 1. Upward trend
            # 2. Bullish Starting candle
            # 3. Gap up the Starting Candle close and trading candidate open
            # 4. Close of trading Candidate less than 50% of the startingCandle
       
        elif t ==1:
            # requirement 1
            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]



            if startingCandle.close > startingCandle.open and tradingCandidate.open > (startingCandle.close + startingCandle.close * 0.001 ):
                        # requirement 2 and 3
                if tradingCandidate.close < ((startingCandle.open + startingCandle.close)/2):
                    # requirement 4
                    print(f'Dark Cloud found in a upward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                    msg = f"Short trade Signal for  {self.contract.symbol} using Dark Cloud Strategy on timeframe {self.tf}"
                    logger.info("%s\nDARK CLOUD TESTING starting candle close %s trading candidate open %s", msg,startingCandle.close,tradingCandidate.open)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return -1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result

## relaxing pattern a little
class PiercingPattern(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            
            '15m':{
                'sl':3,
                'tp':1
            },
            '30m':{
                'sl':3,
                'tp':1
            },     
            '1h':{
                'sl':5,
                'tp':6
                
            },
            '4h':{
                'sl':5,
                'tp':1
            },
            '1d':{
                'sl':5, #23
                'tp':1   #23
            },
            '3d':{
                'sl':5, #20
                'tp':1
            },
            '1w':{
                'sl':5,
                'tp':1
            },
            '1M':{
                'sl':5, #20
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount,  sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Piercing Pattern",'long',averaging_strat = True)
    

    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles =  self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == -1:
                if startingCandle.close < startingCandle.open and tradingCandidate.open < (startingCandle.close - startingCandle.close *0.001):
                    if tradingCandidate.close > ((startingCandle.open + startingCandle.close)/2):
                        
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos + (start_pos * tp_pct)
                        sl_test = start_pos - (start_pos * sl_pct)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.high >= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.low <= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
                    
        return data



    def _check_signal(self) -> int:
 

        t = self.find_trend()

        if t == 0:
            return 0
        elif t ==-1:
            # Requirements for a Piercing Pattern 
            # 1. downward trend
            # 2. Bearish Starting candle
            # 3. Gap down the Starting Candle close and trading candidate open
            # 4. Close of trading Candidate more than 50% of the startingCandle


            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]
            if startingCandle.close < startingCandle.open and tradingCandidate.open < (startingCandle.close - startingCandle.close *0.001):
                if tradingCandidate.close > ((startingCandle.open + startingCandle.close)/2):
                    print(f'Piercing Pattern found in a downtrend trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                    msg = f"Long trade Signal for  {self.contract.symbol} using Piercing Pattern Strategy on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    print("message sent")
                    t = Thread(target=self._write_history).start()
                    return 1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        # print(tick_type,self.stat_name)
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                self._open_position(signal_result)
                pass

        return signal_result

## relaxing pattern a little
class EveningStar(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        sl_tp = {
            '1m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '3m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '5m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '15m':{
                'sl':8,     
                'tp':3
            },
            '30m':{
                'sl':8,      
                'tp':3
            },     
            '1h':{
                'sl':6,    
                'tp':4
            },
            '4h':{
                'sl':9,        
                'tp':1          # No great combination was found going with best 1 and 9        
            },
            '1d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data     
            },
            '3d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
            },
            '1w':{
                'sl':3,        # only one trade was found lets go with 1 and 3 for the time being   
                'tp':1          
            },
            '1M':{
                'sl':9,        # 2 trades were found and hit take profit 25 even for sl 1 going with 5 and 9 for now
                'tp':5          
            },
        }

        super().__init__(client, contract, exchange, timeframe, trade_amount, sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Evening Star",'short')
 
    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            firstCandle  = previousCandles[i-2]
            eveningStar  = previousCandles[i-1]
            tradingCandidate = previousCandles[i]


            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                #if startingCandle.close > startingCandle.open and ((startingCandle.close + startingCandle.open) > ((startingCandle.low + startingCandle.high) * 0.25 )) : 
                if firstCandle.close > firstCandle.open  : 
                    # star
                    if eveningStar.open > eveningStar.close and eveningStar.open > firstCandle.close and ((eveningStar.close + eveningStar.open) <= (abs(eveningStar.low + eveningStar.high)  )) :
                        if tradingCandidate.close < tradingCandidate.open and tradingCandidate.open < eveningStar.close :  
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos - (start_pos * 0.02)
                            sl_test = start_pos + (start_pos * 0.03)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.low <= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.high >= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
        print('printing data' , data)      
        return data


    def _check_signal(self) -> int:
        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        t = self.find_trend()


        if t == 0:
            return 0
            # requirements
            # upward trend  
            # first is a green candle 
            # second will be a small bodied red candle lets say the body must be less than the first bullish candles body
            # and then after that a bearish candle with big body lets say the body must be greater than than the evening star
        elif t ==1:
            # upward trend
            firstCandle  = self.contract.candles[self.tf][-4]
            eveningStar  = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]

            if firstCandle.close > firstCandle.open: 
                    
                if eveningStar.open > eveningStar.close and eveningStar.open > firstCandle.close + (firstCandle.close * 0.001)  and eveningStar.close > tradingCandidate.open + (tradingCandidate.open * 0.001) and tradingCandidate.open > tradingCandidate.close :
                
                    if abs ( eveningStar.close - eveningStar.open ) / eveningStar.open < 0.01:  

                        print(f'Evening Star found in a downtrend trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                        msg = f"Short trade Signal for  {self.contract.symbol} using Evening Star Strategy on timeframe {self.tf}"
                        logger.info("%s", msg)
                        self.send_msg(msg)
                        t = Thread(target=self._write_history).start()
                        return -1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result

## relaxing pattern a little
class MorningStar(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        sl_tp = {
            '1m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '3m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '5m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '15m':{
                'sl':5,     
                'tp':2
            },
            '30m':{
                'sl':4,      
                'tp':1
            },     
            '1h':{
                'sl':17,    
                'tp':8
            },
            '4h':{
                'sl':7,        
                'tp':3                 
            },
            '1d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data     
            },
            '3d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
            },
            '1w':{
                'sl':9,        
                'tp':4          
            },
            '1M':{
                'sl':9,        
                'tp':5          
            },
        }


        super().__init__(client, contract, exchange, timeframe, trade_amount,  sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Morning Star",'long')
    

    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            firstCandle  = previousCandles[i-2]
            morningStar  = previousCandles[i-1]
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)

            if trend == -1:

                if firstCandle.close < firstCandle.open : 
                    # star
                    if morningStar.open < morningStar.close and morningStar.open < firstCandle.close  :
                        if tradingCandidate.close > tradingCandidate.open and tradingCandidate.open > morningStar.close :  
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos + (start_pos * 0.01)
                            sl_test = start_pos - (start_pos * 0.01)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.high >= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.low <= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        print('printing data' , data)      
        return data




    def _check_signal(self) -> int:

        t = self.find_trend()

 

        if t == 0:
            return 0
        elif t ==-1:
            firstCandle  = self.contract.candles[self.tf][-4]
            morningStar  = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]

            if firstCandle.close < firstCandle.open: 
                if morningStar.open <  morningStar.close and morningStar.open  < firstCandle.close - (firstCandle.close * 0.001) and morningStar.close < tradingCandidate.open - (tradingCandidate.open * 0.001):
                    if abs ( morningStar.close - morningStar.open ) / morningStar.open < 0.01:
                        print(f'Morning Star found in a upward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                        msg = f"Long trade Signal for  {self.contract.symbol} using Evening Star Strategy on timeframe {self.tf}"
                        logger.info("%s", msg)
                        self.send_msg(msg)
                        t = Thread(target=self._write_history).start()
                        return 1
        return 0
    
    
    

    def check_trade(self, tick_type: str):
        # two means no change in signal
        self.last_candle = self.contract.candles[self.tf][-1]
          
        
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result






## relaxing pattern a little
class BullishKicker(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Bullish Kicker",'long')
    
    def _check_signal(self) -> int:
 

        t = self.find_trend()


        if t == 0:
            return 0
        elif t ==-1:
            # Downtrend identified 
            # Identify bullish kicker
            # bullish kicker red candle followed by a green candle with a big gap between open of green candle and close of red candle 
            firstCandle = self.contract.candles[self.tf][-3]
            tradingCandidate =self.contract.candles[self.tf][-2] 
            if firstCandle.close < firstCandle.open and (tradingCandidate.open > (firstCandle.close + (firstCandle.close *0.01 ) )) and tradingCandidate.close > tradingCandidate.open:
                print(f'Bullish Kicker Found in a downward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                msg = f"Long trade Signal for  {self.contract.symbol} using {self.stat_name} on timeframe {self.tf}"
                logger.info("%s", msg)
                self.send_msg(msg)
                t = Thread(target=self._write_history).start()
                return 1
        
        return 0
    
    
    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles =  self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == -1:
                if startingCandle.close < startingCandle.open and (tradingCandidate.open > (startingCandle.high + (abs(startingCandle.high + startingCandle.low ) *0.02 ) )) and tradingCandidate.close > startingCandle.close and tradingCandidate.close > tradingCandidate.open:
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos + (start_pos * 0.01)
                        sl_test = start_pos - (start_pos * 0.01)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.high >= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.low <= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
                                        
                    
        return data




    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
          
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result
## relaxing pattern a little
class BearishKicker(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Bearish Kicker",'short')
    
    def _check_signal(self) -> int:

        t = self.find_trend()

        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        if t == 0:
            return 0
        elif t ==1:
            firstCandle = self.contract.candles[self.tf][-3]
            tradingCandidate =self.contract.candles[self.tf][-2] 
            # green candle in upward trend then bearish candle with gap down
            if  firstCandle.close > firstCandle.open and (tradingCandidate.open < (firstCandle.close - (abs(firstCandle.close ) *0.01 ) ))  and tradingCandidate.close < tradingCandidate.open:
                print(f'Bearish Kicker Found in a downward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                msg = f"Short trade Signal for  {self.contract.symbol} using {self.stat_name} on timeframe {self.tf}"
                logger.info("%s", msg)
                self.send_msg(msg)
                t = Thread(target=self._write_history).start()
                return -1
        return 0
    
    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                if startingCandle.close > startingCandle.open and (tradingCandidate.open < (startingCandle.low - (abs(startingCandle.high + startingCandle.low ) *0.02 ) )) and tradingCandidate.close < startingCandle.close and tradingCandidate.close < tradingCandidate.open:
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos - (start_pos * 0.01)
                        sl_test = start_pos + (start_pos * 0.01)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.low <= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.high >= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
                                        
        print('printing data' , data)      
        return data

    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         

        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                #self._open_position(signal_result)
                pass

        return signal_result


# For three black solidiers and three black crows maybe the three candles in downward or upward direction break the trend and trade is not 
# parsed hence no signal is generated 
class ThreeWhiteSolidiers(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        sl_tp = {
            '1m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '3m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '5m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '15m':{
                'sl':3,    
                'tp':1
            },
            '30m':{
                'sl':3,    
                'tp':1
            },     
            '1h':{
                'sl':3,        
                'tp':1
            },
            '4h':{
                'sl':3,        # Results from backtesting were not very good for testing keeping the best combination from backtest        
                'tp':1                  
            },
            '1d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data     
            },
            '3d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
            },
            '1w':{
                'sl':15,         
                'tp':7          
            },
            '1M':{
                'sl':15,        
                'tp':7          
            },
        }

        super().__init__(client, contract, exchange, timeframe, trade_amount,  sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Three White Solidiers",'long')    

    def _check_signal(self) -> int:

        t = self.find_trend()
 


        if t == 0:
            return 0
        elif t ==-1:
            # downward trend identified
            # check for three white soldiers
            firstCandle = self.contract.candles[self.tf][-4]
            secondCandle = self.contract.candles[self.tf][-3]
            tradingCandidate =self.contract.candles[self.tf][-2] 
            
            if firstCandle.close > firstCandle.open and secondCandle.close > firstCandle.close and tradingCandidate.close > secondCandle.close:
                print(f'Three White Solidiers Found in a downward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                msg = f"Long trade Signal for  {self.contract.symbol} using Three White Solidiers Strategy on timeframe {self.tf}"
                logger.info("%s", msg)
                self.send_msg(msg)
                t = Thread(target=self._write_history).start()
                return 1
        return 0
    
    def backtest(self,contract:Contract,timeframe:str):
        print("three white solidiers backtest")
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            firstCandle = previousCandles[i-2]
            secondCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)

            if trend == -1:
                if firstCandle.close > firstCandle.open and secondCandle.close > firstCandle.close and tradingCandidate.close > secondCandle.close:
                            print(f"downward trend for three white solidiers and first candle {firstCandle.timestamp} : {firstCandle.close} ,secondCandle candle {secondCandle.timestamp} : {secondCandle.close},tradingCandidate candle {tradingCandidate.timestamp} : {tradingCandidate.close} ")
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos + (start_pos * 0.01)
                            sl_test = start_pos - (start_pos * 0.01)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.high >= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.low <= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        # print('printing data' , data)      
        return data



    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result

class ThreeBlackCrows(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        sl_tp = {
            '1m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '3m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '5m':{
                'sl':1,     # Not useful
                'tp':1      # Not useful
            },
            '15m':{
                'sl':8,     
                'tp':3
            },
            '30m':{
                'sl':3,    # No good combination found going  
                'tp':1
            },     
            '1h':{
                'sl':3,    # No trade or combination was profitable 
                'tp':1
            },
            '4h':{
                'sl':3,        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
            },
            '1d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data     
            },
            '3d':{
                'sl':3,         # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
                'tp':1          # No trades were found in backtesting so keeping at 3 and 1 will update with new data        
            },
            '1w':{
                'sl':13,        
                'tp':5          
            },
            '1M':{
                'sl':10,        
                'tp':5          
            },
        }

        super().__init__(client, contract, exchange, timeframe, trade_amount, sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Three Black Crows",'short')    

    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            firstCandle = previousCandles[i-2]
            secondCandle = previousCandles[i-1]
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close

            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                if firstCandle.close < firstCandle.open and secondCandle.close < firstCandle.close and tradingCandidate.close < secondCandle.close:
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos - (start_pos * 0.01)
                            sl_test = start_pos + (start_pos * 0.01)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.low <= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.high >= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        print('printing data' , data)      
        return data




    def _check_signal(self) -> int:

        t = self.find_trend()
        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)


        if t == 0:
            return 0
        elif t ==1:
            # upward trend identified
            # check for three black crows
            firstCandle = self.contract.candles[self.tf][-4]
            secondCandle = self.contract.candles[self.tf][-3]
            tradingCandidate =self.contract.candles[self.tf][-2] 
            if firstCandle.close < firstCandle.open and secondCandle.close < firstCandle.close and tradingCandidate.close < secondCandle.close:
                print(f'Three Black Crows Found in a upward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                msg = f"Short trade Signal for  {self.contract.symbol} using Three Black Crows Strategy on timeframe {self.tf}"
                logger.info("%s", msg)
                self.send_msg(msg)
                t = Thread(target=self._write_history).start()
                return -1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                #self._open_position(signal_result)
                pass

        return signal_result


## Cannot be realxed further maybe changing the trend definition has helped
class BullishEngulfing(Strategy):

    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            
            '15m':{
                'sl':3,
                'tp':1
            },
            '30m':{
                'sl':3,
                'tp':1
            },     
            '1h':{
                'sl':25,
                'tp':15
            },
            '4h':{
                'sl':8,
                'tp':2
            },
            '1d':{
                'sl':20,
                'tp':10
            },
            '3d':{
                'sl':15,
                'tp':7
            },
            '1w':{
                'sl':3,
                'tp':1
            },
            '1M':{
                'sl':3,
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount, sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Bullish Engulfing",'long')    


    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle  = previousCandles[i-1]
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)

            if trend == -1:
                if startingCandle.open > startingCandle.close:
                    # fIRST CANDLE IS Red
                    if tradingCandidate.open < tradingCandidate.close and (tradingCandidate.high > startingCandle.high and tradingCandidate.low < startingCandle.low and tradingCandidate.close > startingCandle.close):
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos + (start_pos * 0.01)
                            sl_test = start_pos - (start_pos * 0.01)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.high >= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.low <= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        print('printing data' , data)      
        return data





    def _check_signal(self) -> int:

        t = self.find_trend()

 

        if t == 0:
            return 0
        elif t ==-1:
            """
                A bullish engulfing candlestick pattern occurs at the end of a downtrend. It consists of two candles, with the first candle having a relatively small body and short shadows, also known as wicks. The second candle, on the other hand, has longer wicks and a real body that engulfs the body of the previous candle.
            """
            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]
            # downward trend identified
            # check for engulfing
            if startingCandle.open > startingCandle.close:
                # fIRST CANDLE IS Red
                if tradingCandidate.open < tradingCandidate.close and (tradingCandidate.high > startingCandle.high and tradingCandidate.low < startingCandle.low and tradingCandidate.close > startingCandle.open):
                    #trading candidate is green and engulfs the red candle
                    print(f'Bullish Engulfing Pattern Found in a upward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                    msg = f"Long trade Signal for  {self.contract.symbol} using Bullish Engulfing Strategy on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return 1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result

## Cannot be realxed further maybe changing the trend definition has helped
class BearishEngulfing(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
    
        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            
            '15m':{
                'sl':3,
                'tp':1
            },
            '30m':{
                'sl':3,
                'tp':1
            },     
            '1h':{
                'sl':1,
                'tp':1
            },
            '4h':{
                'sl':7,
                'tp':3
            },
            '1d':{
                'sl':7,
                'tp':1
            },
            '3d':{
                'sl':5,
                'tp':1
            },
            '1w':{
                'sl':8,
                'tp':4
            },
            '1M':{
                'sl':1,
                'tp':1
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount,  sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Bearish Engulfing",'short')    

    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            startingCandle  = previousCandles[i-1]
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:
                if startingCandle.close > startingCandle.open:
 
                    if tradingCandidate.close < tradingCandidate.open and (tradingCandidate.high > startingCandle.high and tradingCandidate.low < startingCandle.low and tradingCandidate.close < startingCandle.close):
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos - (start_pos * 0.01)
                            sl_test = start_pos + (start_pos * 0.01)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.low <= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.high >= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        print('printing data' , data)      
        return data



    def _check_signal(self) -> int:

        t = self.find_trend()

        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        if t == 0:
            return 0
        elif t ==1:
            startingCandle = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]
            
            if startingCandle.close > startingCandle.open:
                # first candle is green
                # now the trading candidate must be red an fully engulf the bullish candle
                if tradingCandidate.close < tradingCandidate.open and (tradingCandidate.high > startingCandle.high and tradingCandidate.low < startingCandle.low and tradingCandidate.close < startingCandle.open):
                    #trading candle is red
             
                    print(f'Bearish Engulfing Pattern Found in a upward trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                    msg = f"Short trade Signal for  {self.contract.symbol} using Bearish Engulfing Strategy on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return 1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                #self._open_position(signal_result)
                pass

        return signal_result

# bullish continuation pattern
## relaxing pattern a little
class RisingThree(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Rising Three",'long')    



    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles =  self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            tradingCandidate = previousCandles[i]
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-2].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == 1:

                firstBullish = previousCandles[i-4]
                firstFall = previousCandles[i-3]
                secondFall = previousCandles[i-2]
                thirdFall = previousCandles[i-1]
                tradingCandidate = previousCandles[i]
                if firstBullish.close > firstBullish.open:
                    if firstFall.close < firstFall.open and firstFall.low > firstBullish.open and secondFall.close < firstFall.close and secondFall.low > firstBullish.open  and thirdFall.close < secondFall.close and thirdFall.low >  firstBullish.open and tradingCandidate.close > firstFall.open:
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos + (start_pos * 0.01)
                        sl_test = start_pos - (start_pos * 0.01)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.high >= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.low <= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
                    
        return data



    def _check_signal(self) -> int:
        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        t = self.find_trend()


        if t == 0:
            return 0
        elif t == 1:
           # bullish candle followed by 3 bearish candles and then another bullish candle 
           # the bearish candles will not fall below the open of the first bullish candle
           # the last bullish candle will have greater close than first bullish candle  
            firstBullish = self.contract.candles[self.tf][-6]
            firstFall = self.contract.candles[self.tf][-5]
            secondFall = self.contract.candles[self.tf][-4]
            thirdFall = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]

            if firstBullish.close > firstBullish.open:
                if firstFall.close < firstFall.open and firstFall.close > firstBullish.open and secondFall.close < secondFall.open and secondFall.close > firstBullish.open  and thirdFall.close < thirdFall.open and thirdFall.close >  firstBullish.open and tradingCandidate.close > firstBullish.close:
                    print(f'Rising three Pattern Found in a uptrend trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                    msg = f"long trade Signal for  {self.contract.symbol} using Rising Three Strategy on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return 1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                # self._open_position(signal_result)
                pass

        return signal_result

# Bearish continuation pattern
## relaxing pattern a little
class FallingThree(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "Falling Three",'short')    



    def _check_signal(self) -> int:
 

        t = self.find_trend()


        if t == 0:
            return 0
        elif t == -1:
            # downtrend identified
            # first bearish followed by three bullish all with close less than first bearish open finally another bearish with close lower than first bearish
            firstBearish = self.contract.candles[self.tf][-6]
            firstRise = self.contract.candles[self.tf][-5]
            secondRise = self.contract.candles[self.tf][-4]
            thirdRise = self.contract.candles[self.tf][-3]
            tradingCandidate = self.contract.candles[self.tf][-2]
            
            if firstBearish.close < firstBearish.open:
                if firstRise.close > firstRise.open and firstRise.close < firstBearish.open and secondRise.close > secondRise.open and secondRise.close < firstBearish.open  and thirdRise.close > thirdRise.open and thirdRise.close <  firstBearish.open and tradingCandidate.close < firstRise.close:
                    print(f'Falling Three Pattern Found in a downwarf trend for {self.contract.symbol} on timeframe {self.tf} ')                            
                    msg = f"Short trade Signal for  {self.contract.symbol} using Falling Three Strategy on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return -1
        return 0
    


    def backtest(self,contract:Contract,timeframe:str):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)

        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            firstBearish = previousCandles[i-4]
            firstRise = previousCandles[i-3]
            secondRise = previousCandles[i-2]
            thirdRise = previousCandles[i-1]
            tradingCandidate = previousCandles[i]
            
            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)
            if trend == -1:
                print('trend is down')
                if firstBearish.close < firstBearish.open:
                    print("first candle is bearish")
                    if firstRise.close > firstRise.open and firstRise.high < firstBearish.open and secondRise.close > firstRise.close and secondRise.high < firstBearish.open  and thirdRise.close > secondRise.close and thirdRise.high <  firstBearish.open and tradingCandidate.close < firstRise.open:                        
                        print(f'Trade found {timeframe} , {self.contract.symbol}')
                        start_pos = tradingCandidate.close
                        tp_test = start_pos - (start_pos * 0.01)
                        sl_test = start_pos + (start_pos * 0.01)
                        j = 1
                        obj = {
                            'start_pos': start_pos,
                            'tp_test': tp_test,
                            'sl_test': sl_test,
                            'enterTimestamp' : tradingCandidate.timestamp,
                            'symbol': self.contract.symbol,
                            'tf': timeframe
                        }
                        print('start_pos',start_pos)
                        print('tp_test',tp_test)
                        print('sl_test',sl_test)
                        print('enterTimestamp',tradingCandidate.timestamp)
                        while True:
                            try:
                                testCandle = previousCandles[i+j]
                                obj['finalcandleTimestamp'] =  None
                                obj['finalcandleClose'] = None
                                obj['tp_hit'] = None
                                obj['sl_hit'] = None
                                obj['finalcandleHigh'] = None
                                obj['finalcandleLow'] = None
                                obj['finalcandleOpen'] = None
                                if testCandle.low <= tp_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.low
                                    obj['tp_hit'] = True
                                    obj['sl_hit'] = False
                                    data.append(obj)
                                    break
                                elif testCandle.high >= sl_test:
                                    obj['exitTimestamp'] = testCandle.timestamp
                                    obj['candleForward'] = j
                                    obj['exit_pos'] = testCandle.high
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = True
                                    data.append(obj)
                                    break
                                j = j + 1
                            except IndexError as e:
                                print('End of candles but stoploss or tp not hit yet')
                                testCandle = previousCandles[i]
                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                obj['finalcandleClose'] = testCandle.close
                                obj['tp_hit'] = False
                                obj['sl_hit'] = False
                                obj['finalcandleHigh'] = testCandle.high
                                obj['finalcandleLow'] = testCandle.low
                                obj['finalcandleOpen'] = testCandle.open
                                data.append(obj)
                                break
                                        
       
        return data




    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                #self._open_position(signal_result)
                pass

        return signal_result

class A_SELL_01(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        """This will take timeframe like the rest of the strategies but overwrite timeframe to 1"""
        timeframe = '1h'
        # we will use get historical data to get the latest candle for 4 hourly and daily candle
        # get latest 
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "A_SELL_01",'short')    


    def backTest(self,contract:Contract):
        self.contract = contract 
        obj = {}
        tp_test = 0
        sl_test = 0
        start_pos = 0
        data = []
        dailyCandles = self.client.get_historical_candles(self.contract, "1d",limit=50)
        
        for daily in dailyCandles:
            if daily.close < daily.open:
                print("daily candle is red")
                fourHourlyCandles = self.client.get_historical_candles_test(daily.timestamp,daily.timestamp+86400000,self.contract, "4h")
                for fourHourly in fourHourlyCandles:
                    if fourHourly.close < fourHourly.open:
                        print("4 hourly candle is red")
                        hourly = self.client.get_historical_candles_test(fourHourly.timestamp,fourHourly.timestamp+(14400000 ),self.contract, "1h")
                        for i in range (len(hourly)-1):   
                            hourly2 = hourly[i]
                            hourly1 = hourly[i+1]     
                            if not (hourly2.close < hourly2.open and hourly1.close < hourly1.open):
                                continue
                            if hourly2.low  >=  (hourly2.close - (abs(hourly2.open - hourly2.close) *0.50)):
                                if hourly1.open <= (hourly2.open - abs((hourly2.open - hourly2.close) *0.50)):
                                    print(f'trade found' )
                                    start_pos = hourly1.close
                                    tp_test = start_pos - (start_pos * 0.01)
                                    sl_test = hourly2.high
                                    j = 1
                                    obj = {
                                        'start_pos': start_pos,
                                        'tp_test': tp_test,
                                        'sl_test': sl_test,
                                        'enterTimestamp' : hourly1.timestamp,
                                        'symbol': self.contract.symbol,
                                    }
                                    obj['Daily_Timestamp'] = daily.timestamp
                                    obj['FourHourly_Timestamp'] = fourHourly.timestamp
                                    obj['FirstHourly_Timestamp'] = hourly2.timestamp 

                                    nextCandles = self.client.get_historical_candles_test(hourly1.timestamp, float('inf'),self.contract,'1h')
                                    i = 0
                                    while True:
                                        try:
                                            testCandle = nextCandles[i]
                                            obj['finalcandleTimestamp'] =  None
                                            obj['finalcandleClose'] = None
                                            obj['tp_hit'] = None
                                            obj['sl_hit'] = None
                                            obj['finalcandleHigh'] = None
                                            obj['finalcandleLow'] = None
                                            obj['finalcandleOpen'] = None
                                            if testCandle.low <= tp_test:
                                                # take profit hit at j next candle than candidate candle
                                                obj['exitTimestamp'] = testCandle.timestamp
                                                obj['candleForward'] = i
                                                obj['exit_pos'] = testCandle.low
                                                obj['tp_hit'] = True
                                                obj['sl_hit'] = False
                                                break
                                            elif testCandle.high >= sl_test:
                                                obj['exitTimestamp'] = testCandle.timestamp
                                                obj['candleForward'] = j
                                                obj['exit_pos'] = testCandle.high
                                                obj['tp_hit'] = False
                                                obj['sl_hit'] = True
                                                break
                                            i = i + 1
                                        except IndexError as e:
                                            print('End of candles but stoploss or tp not hit yet')
                                            testCandle = nextCandles[i-1]
                                            obj['finalcandleTimestamp'] =  testCandle.timestamp
                                            obj['finalcandleClose'] = testCandle.close
                                            obj['tp_hit'] = False
                                            obj['sl_hit'] = False
                                            obj['finalcandleHigh'] = testCandle.high
                                            obj['finalcandleLow'] = testCandle.low
                                            obj['finalcandleOpen'] = testCandle.open
                                            break
                                    data.append(obj)
                                    
            
       
        return data
    def _check_signal(self) -> int:

        latestDaily = self.client.get_historical_candles(self.contract,'1d',1)[0]
        latestFourHour =  self.client.get_historical_candles(self.contract,'4h',1)[0]
        # tail of first hourly candle must be less than 100% of the body
        if latestDaily.close < latestDaily.open:
            print("latest Daily candle is red")
            if latestFourHour.close < latestFourHour.open:
                print("latest 4 hour candle is red")
                if self.contract.candles[self.tf][-3].close < self.contract.candles[self.tf][-3].open and self.contract.candles[self.tf][-2].close < self.contract.candles[self.tf][-2].open:
                    # both candles are red
                    if self.contract.candles[self.tf][-3].low  >=  (self.contract.candles[self.tf][-3].close - (abs(self.contract.candles[self.tf][-3].open - self.contract.candles[self.tf][-3].close) *0.50)):
                        if self.contract.candles[self.tf][-2].open <= (self.contract.candles[self.tf][-3].open - abs((self.contract.candles[self.tf][-3].open - self.contract.candles[self.tf][-3].close) *0.50)):
                            print(f'short trade signal using A_01 Stratergy for {self.contract.symbol} on timeframe{self.tf} ')                            
                            msg = f"Short trade Signal for  {self.contract.symbol} using A_01 Stratergy on timeframe {self.tf}"
                            logger.info("%s", msg)
                            self.send_msg(msg)
                return -1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                #self._open_position(signal_result)
                pass

        return signal_result

class A_BUY_01(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        # print(client.get_historical_candles)
        """This will take timeframe like the rest of the strategies but overwrite timeframe to 1"""
        timeframe = '1h'
        print("init strat a_01 buy")
        # we will use get historical data to get the latest candle for 4 hourly and daily candle
        # get latest 
        super().__init__(client, contract, exchange, timeframe, trade_amount, take_profit, stop_loss, "A_BUY_01",'long')    


    def backTest(self,contract : Contract):
        self.contract = contract
        dailyCandles = self.client.get_historical_candles(self.contract, "1d",limit=1000)
        data = []
        for daily in dailyCandles:
            if daily.close > daily.open:
                fourHourlyCandles = self.client.get_historical_candles_test(daily.timestamp,daily.timestamp+86400000,self.contract, "4h")
                for fourHourly in fourHourlyCandles:
                    if fourHourly.close > fourHourly.open:
                        hourly = self.client.get_historical_candles_test(fourHourly.timestamp,fourHourly.timestamp+(14400000 ),self.contract, "1h")
                        for i in range (len(hourly)-1):  
                            hourly2 = hourly[i]
                            hourly1 = hourly[i+1]
                            if hourly2.close > hourly2.open and hourly1.close > hourly1.open:
                                if hourly2.high  <=  (hourly2.close + (abs(hourly2.open - hourly2.close) *0.50)):
                                    if hourly1.open >= (hourly2.open + abs((hourly2.open - hourly2.close) *0.50)):
                                        print(f'trade found' )

                                        start_pos = hourly1.close
                                        tp_test = start_pos + (start_pos * 0.01)
                                        sl_test = hourly2.low
                                        j = 1
                                        obj = {
                                            'start_pos': start_pos,
                                            'tp_test': tp_test,
                                            'sl_test': sl_test,
                                            'enterTimestamp' : hourly1.timestamp,
                                            'symbol': self.contract.symbol,
                                        }
                                        obj['Daily_Timestamp'] = daily.timestamp
                                        obj['FourHourly_Timestamp'] = fourHourly.timestamp
                                        obj['FirstHourly_Timestamp'] = hourly2.timestamp 
                                        nextCandles = self.client.get_historical_candles_test(hourly1.timestamp, float('inf'),self.contract,'1h')
                                        i = 0
                                        while True:
                                            try:
                                                testCandle = nextCandles[i]
                                                obj['finalcandleTimestamp'] =  None
                                                obj['finalcandleClose'] = None
                                                obj['tp_hit'] = None
                                                obj['sl_hit'] = None
                                                obj['finalcandleHigh'] = None
                                                obj['finalcandleLow'] = None
                                                obj['finalcandleOpen'] = None
                                                if testCandle.high >= tp_test:
                                                    # take profit hit at j next candle than candidate candle
                                                    obj['exitTimestamp'] = testCandle.timestamp
                                                    obj['candleForward'] = i
                                                    obj['exit_pos'] = testCandle.low
                                                    obj['tp_hit'] = True
                                                    obj['sl_hit'] = False
                                                    break
                                                elif testCandle.low <= sl_test:
                                                    obj['exitTimestamp'] = testCandle.timestamp
                                                    obj['candleForward'] = j
                                                    obj['exit_pos'] = testCandle.high
                                                    obj['tp_hit'] = False
                                                    obj['sl_hit'] = True
                                                    break
                                                i = i + 1
                                            except IndexError as e:
                                                print('End of candles but stoploss or tp not hit yet')
                                                testCandle = nextCandles[i-1]
                                                obj['finalcandleTimestamp'] =  testCandle.timestamp
                                                obj['finalcandleClose'] = testCandle.close
                                                obj['tp_hit'] = False
                                                obj['sl_hit'] = False
                                                obj['finalcandleHigh'] = testCandle.high
                                                obj['finalcandleLow'] = testCandle.low
                                                obj['finalcandleOpen'] = testCandle.open
                                                break
                                        data.append(obj)
                
                    continue
            continue

        return data
    def _check_signal(self) -> int:



        latestDaily = self.client.get_historical_candles(self.contract,'1d',1)[0]
        latestFourHour =  self.client.get_historical_candles(self.contract,'4h',1)[0]
        # tail of first hourly candle must be less than 100% of the body
        if latestDaily.close > latestDaily.open:
            print("latest Daily candle is green")
            if latestFourHour.close > latestFourHour.open:
                print("latest 4 hour candle is green")
                if self.contract.candles[self.tf][-3].close > self.contract.candles[self.tf][-3].open and self.contract.candles[self.tf][-2].close > self.contract.candles[self.tf][-2].open:
                    # both candles are Green
                    if self.contract.candles[self.tf][-3].high  <=  (self.contract.candles[self.tf][-3].close + (abs(self.contract.candles[self.tf][-3].close - self.contract.candles[self.tf][-3].open) *0.50)):
                        if self.contract.candles[self.tf][-2].open >= (self.contract.candles[self.tf][-3].open + abs((self.contract.candles[self.tf][-3].close - self.contract.candles[self.tf][-3].open) *0.50)):
                            print(f'long trade signal using A_02 Stratergy for {self.contract.symbol} on timeframe{self.tf} ')                            
                            msg = f"Long trade Signal for  {self.contract.symbol} using A_02 Stratergy on timeframe {self.tf}"
                            logger.info("%s", msg)
                            self.send_msg(msg)
                return 1
        return 0
    
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                #self._open_position(signal_result)
                pass

        return signal_result

## relaxing pattern a little
class LongWickBullish(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,

                 stop_loss: float):
        """
            Long wick bullish is a reversal pattern the long tail on the lower side indicates that seller tried to drag dowwn the price but buyers presisted 
            indicating a trend reversal check for a long tailed candle in a bearish market
            """
        sl_tp = {
            '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            '15m':{
                'sl':1,
                'tp':1
            },
            '30m':{
                'sl':1,
                'tp':1
            },     
            '1h':{
                'sl':1,
                'tp':1
            },
            '4h':{
                'sl':1,
                'tp':1.5
            },
            '1d':{
                'sl':1,
                'tp':2
            },
            '3d':{
                'sl':1,
                'tp':2
            },
            '1w':{
                'sl':1,
                'tp':2
            },
            '1M':{
                'sl':1,
                'tp':2
            },

        }
  
        super().__init__(client, contract, exchange, timeframe, trade_amount,sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Long Wick Bullish",'long')
    

    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)

            if trend == -1:
                if tradingCandidate.close > tradingCandidate.open:
                    # candle is green check if it has a long tail lets say 150% of the body 
                    if abs(tradingCandidate.open - tradingCandidate.close) *1.5 < tradingCandidate.open -tradingCandidate.low:
                        # check if the trading candidate low is lowest in the previous 5 candles
                        prev1 = previousCandles[i-1].low
                        prev2 = previousCandles[i-2].low
                        prev3 = previousCandles[i-3].low
                        prev4 = previousCandles[i-4].low
                        prev5 = previousCandles[i-5].low
                        if min(prev1,prev2,prev3,prev4,prev5) > tradingCandidate.low:
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos + (start_pos *  tp_pct)
                            sl_test = start_pos - (start_pos *  sl_pct)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.high >= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.low <= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
                                            
        print('printing data' , data)      
        return data






    def _check_signal(self) -> int:
        t = self.find_trend()
 

        if t == 0:
            return 0
        elif t ==-1:           
            tradingCandidate = self.contract.candles[self.tf][-2]

            if tradingCandidate.close > tradingCandidate.open:
                lowerWick = abs(tradingCandidate.open  - tradingCandidate.low)
                upperWick = abs(tradingCandidate.close - tradingCandidate.high)
                if abs(tradingCandidate.close - tradingCandidate.open) * 8 < lowerWick  and upperWick *2 < lowerWick :
                    msg = f"Long trade Signal for  {self.contract.symbol} using {self.stat_name} on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return 1
        return 0
    




    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        # two means no change in signal
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                if self.tf not in ['15m', '30m']:
                    self._open_position(signal_result,self.contract.candles[self.tf][-2].low)
                

        return signal_result



## relaxing pattern a little
class LongWickBearish(Strategy):
    def __init__(self, client, contract: Contract, exchange: str, timeframe: str, trade_amount: float, take_profit: float,
                 stop_loss: float):
        """
            Long wick bearish is a reversal pattern the long tail on the upper side indicates that buyers tried to increase the price but sellers presisted 
            indicating a trend reversal check for a long tailed candle in a bullish market
            """
        sl_tp = {
        '1m':{
                'sl':1,
                'tp':1
            },
            '3m':{
                'sl':1,
                'tp':1
            },
            '5m':{
                'sl':1,
                'tp':1
            },
            '15m':{
                'sl':1,
                'tp':1
            },
            '30m':{
                'sl':1,
                'tp':1
            },     
            '1h':{
                'sl':1,
                'tp':1
            },
            '4h':{
                'sl':1,
                'tp':1
            },
            '1d':{
                'sl':1,
                'tp':2
            },
            '3d':{
                'sl':1,
                'tp':3
            },
            '1w':{
                'sl':1,
                'tp':3
            },
            '1M':{
                'sl':1,
                'tp':3
            },
        }
        super().__init__(client, contract, exchange, timeframe, trade_amount, sl_tp[timeframe]['tp'], sl_tp[timeframe]['sl'], "Long Wick Bearish",'short')
 
    def backtest(self,contract:Contract,timeframe:str,tp_pct,sl_pct):
        self.contract = contract
        previousCandles = self.client.get_historical_candles(self.contract, timeframe, limit=1000)
        print(len(previousCandles))
        data = []
        tp_test = 0
        sl_test = 0
        start_pos = 0
        for i in range(18, len(previousCandles)):
            tradingCandidate = previousCandles[i]

            trendStart = previousCandles[i-18].close
            trendEnd = previousCandles[i-3].close
            trend = self.find_trend_test(trendStart,trendEnd)

            if trend == 1:
                if tradingCandidate.close < tradingCandidate.open:
                    # candle is red check if it has a long head lets say 150% of the body 
                    if abs(tradingCandidate.open - tradingCandidate.close) * 1.5 < tradingCandidate.high -tradingCandidate.open:
                        # check if the trading candidate low is lowest in the previous 5 candles
                        prev1 = previousCandles[i-1].high
                        prev2 = previousCandles[i-2].high
                        prev3 = previousCandles[i-3].high
                        prev4 = previousCandles[i-4].high
                        prev5 = previousCandles[i-5].high
                        if max(prev1,prev2,prev3,prev4,prev5) < tradingCandidate.high:
                            print(f'Trade found {timeframe} , {self.contract.symbol}')
                            start_pos = tradingCandidate.close
                            tp_test = start_pos - (start_pos *  tp_pct)
                            sl_test = start_pos + (start_pos *  sl_pct)
                            j = 1
                            obj = {
                                'start_pos': start_pos,
                                'tp_test': tp_test,
                                'sl_test': sl_test,
                                'enterTimestamp' : tradingCandidate.timestamp,
                                'symbol': self.contract.symbol,
                                'tf': timeframe
                            }
                            print('start_pos',start_pos)
                            print('tp_test',tp_test)
                            print('sl_test',sl_test)
                            print('enterTimestamp',tradingCandidate.timestamp)
                            while True:
                                try:
                                    testCandle = previousCandles[i+j]
                                    obj['finalcandleTimestamp'] =  None
                                    obj['finalcandleClose'] = None
                                    obj['tp_hit'] = None
                                    obj['sl_hit'] = None
                                    obj['finalcandleHigh'] = None
                                    obj['finalcandleLow'] = None
                                    obj['finalcandleOpen'] = None
                                    if testCandle.low <= tp_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.low
                                        obj['tp_hit'] = True
                                        obj['sl_hit'] = False
                                        data.append(obj)
                                        break
                                    elif testCandle.high >= sl_test:
                                        obj['exitTimestamp'] = testCandle.timestamp
                                        obj['candleForward'] = j
                                        obj['exit_pos'] = testCandle.high
                                        obj['tp_hit'] = False
                                        obj['sl_hit'] = True
                                        data.append(obj)
                                        break
                                    j = j + 1
                                except IndexError as e:
                                    print('End of candles but stoploss or tp not hit yet')
                                    testCandle = previousCandles[i]
                                    obj['finalcandleTimestamp'] =  testCandle.timestamp
                                    obj['finalcandleClose'] = testCandle.close
                                    obj['tp_hit'] = False
                                    obj['sl_hit'] = False
                                    obj['finalcandleHigh'] = testCandle.high
                                    obj['finalcandleLow'] = testCandle.low
                                    obj['finalcandleOpen'] = testCandle.open
                                    data.append(obj)
                                    break
        print('printing data' , data)      
        return data





    def _check_signal(self) -> int:

        t = self.find_trend()

        logger.info("CHECKING SIGNAL FOR %s %s ",  self.contract.symbol, self.stat_name)

        if t == 0:
            return 0
        
        elif t ==1:
            tradingCandidate = self.contract.candles[self.tf][-2]

            if tradingCandidate.close < tradingCandidate.open:
                lowerWick = abs(tradingCandidate.close  - tradingCandidate.low)
                upperWick = abs(tradingCandidate.open - tradingCandidate.high)
                

                if abs(tradingCandidate.close - tradingCandidate.open) * 8 < upperWick and lowerWick * 2 < upperWick:
                    msg = f"Short trade Signal for  {self.contract.symbol} using {self.stat_name} on timeframe {self.tf}"
                    logger.info("%s", msg)
                    self.send_msg(msg)
                    t = Thread(target=self._write_history).start()
                    return -1
        return 0
    
    
    def check_trade(self, tick_type: str):
        self.last_candle = self.contract.candles[self.tf][-1]
         
        signal_result = 2
        if tick_type == "new_candle" and not self.ongoing_position:
            signal_result = self._check_signal()

            if signal_result in [1, -1]:
                if self.tf not in ['15m', '30m']:
                    self._open_position(signal_result,self.contract.candles[self.tf][-2].high)

        return signal_result

