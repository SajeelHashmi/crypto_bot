import dateutil.parser
import datetime
import typing
import pandas as pd 
import numpy as np
# from emailAlerts import sendEmail

BITMEX_MULTIPLIER = 0.00000001
BITMEX_TF_MINUTES = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}


class Balance:
    def __init__(self, info, exchange):
        if exchange == "binance":
            self.initial_margin = float(info['initialMargin'])
            self.maintenance_margin = float(info['maintMargin'])
            self.margin_balance = float(info['marginBalance'])
            self.wallet_balance = float(info['walletBalance'])
            self.unrealized_pnl = float(info['unrealizedProfit'])

        elif exchange == "bitmex":
            self.initial_margin = info['initMargin'] * BITMEX_MULTIPLIER
            self.maintenance_margin = info['maintMargin'] * BITMEX_MULTIPLIER
            self.margin_balance = info['marginBalance'] * BITMEX_MULTIPLIER
            self.wallet_balance = info['walletBalance'] * BITMEX_MULTIPLIER
            self.unrealized_pnl = info['unrealisedPnl'] * BITMEX_MULTIPLIER


class Candle:
    def __init__(self, candle_info, timeframe, exchange):
        if exchange == "binance":
            self.timestamp = candle_info[0]
            self.open = float(candle_info[1])
            self.high = float(candle_info[2])
            self.low = float(candle_info[3])
            self.close = float(candle_info[4])
            self.volume = float(candle_info[5])
            self.timeframe = timeframe
            self.rsi = 0
        if exchange =='test':
            self.timestamp = candle_info['timestamp']
            self.open = float(candle_info['open'])
            self.high = float(candle_info['high'])
            self.low = float(candle_info['low'])
            self.close = float(candle_info['close'])
            self.volume = float(candle_info['volume'])

        if exchange =='testRsi':
            self.timestamp = candle_info['timestamp']
            self.open = float(candle_info['open'])
            self.high = float(candle_info['high'])
            self.low = float(candle_info['low'])
            self.close = float(candle_info['close'])
            self.volume = float(candle_info['volume'])
            self.rsi = float(candle_info['RSI'])
            self.volMa = float(candle_info['volMa'])
  

            
        elif exchange == "bitmex":
            self.timestamp = dateutil.parser.isoparse(candle_info['timestamp'])
            self.timestamp = self.timestamp - datetime.timedelta(minutes=BITMEX_TF_MINUTES[timeframe])
            self.timestamp = int(self.timestamp.timestamp() * 1000)
            self.open = candle_info['open']
            self.high = candle_info['high']
            self.low = candle_info['low']
            self.close = candle_info['close']
            self.volume = candle_info['volume']

        elif exchange == "parse_trade":
            self.timestamp =    float(candle_info['ts'])
            self.open =         float(candle_info['open'])
            self.high =     float(candle_info['high'])
            self.low =  float(candle_info['low'])
            self.close =    float(candle_info['close'])
            self.volume =   float(candle_info['volume'])
            self.rsi = 0




        
def tick_to_decimals(tick_size: float) -> int:
    tick_size_str = "{0:.8f}".format(tick_size)
    while tick_size_str[-1] == "0":
        tick_size_str = tick_size_str[:-1]

    split_tick = tick_size_str.split(".")

    if len(split_tick) > 1:
        return len(split_tick[1])
    else:
        return 0


class Contract:
    def __init__(self, contract_info, exchange):
        if exchange == "binance":
            self.symbol = contract_info['symbol']
            self.base_asset = contract_info['baseAsset']
            self.quote_asset = contract_info['quoteAsset']
            self.price_decimals = contract_info['pricePrecision']
            self.quantity_decimals = contract_info['quantityPrecision']
            self.tick_size = 1 / pow(10, contract_info['pricePrecision'])
            for f in contract_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    self.lot_size = float(f['stepSize'])
            self.candlesRetrieved:typing.Dict[str,bool] = {"1m":False,"3m":False,"5m":False,"15m":False,"30m":False,"1h":False,"4h":False,"1d":False,"3d":False,"1w":False,"1M":False}
            self.candles:typing.Dict[str,typing.List[Candle]] = {}

        elif exchange == "bitmex":
            self.symbol = contract_info['symbol']
            self.base_asset = contract_info['rootSymbol']
            self.quote_asset = contract_info['quoteCurrency']
            self.price_decimals = tick_to_decimals(contract_info['tickSize'])
            self.quantity_decimals = tick_to_decimals(contract_info['lotSize'])
            self.tick_size = contract_info['tickSize']
            self.lot_size = contract_info['lotSize']

            self.quanto = contract_info['isQuanto']
            self.inverse = contract_info['isInverse']

            self.multiplier = contract_info['multiplier'] * BITMEX_MULTIPLIER

            if self.inverse:
                self.multiplier *= -1

        self.exchange = exchange


    def append_candle(self, candle,tf):
        self.candles[tf].append(candle)
        l = len(self.candles)
        if l > 2000:
            self.candles[tf] = self.candles[tf][-1000:]


class OrderStatus:
    def __init__(self, order_info, exchange):
        if exchange == "binance":
            self.order_id = order_info['orderId']
            self.status = order_info['status'].lower()
            self.avg_price = float(order_info['avgPrice'])
        elif exchange == "bitmex":
            self.order_id = order_info['orderID']
            self.status = order_info['ordStatus'].lower()
            self.avg_price = order_info['avgPx']


class Trade:
    def __init__(self, trade_info):
        self.time: int = trade_info['time']
        self.timeframe :str = trade_info['timeframe'] 
        self.contract: Contract = trade_info['contract']
        self.strategy: str = trade_info['strategy']
        self.side: str = trade_info['side']
        self.entry_price: float = trade_info['entry_price']
        self.start_price = self.entry_price
        self.status: str = trade_info['status']
        self.pnl: float = trade_info['pnl']
        self.quantity = trade_info['quantity']
        self.entry_id = trade_info['entry_id']
        self.sl_order_id = trade_info['sl_order_id']
        self.tp_order_id = trade_info['tp_order_id']

        self.avg_price = self.entry_price
        
        self.averaged = False
        
        self.id = f"{self.contract.symbol}_{self.strategy}_{self.side}_{self.timeframe}_{datetime.datetime.now()}"
        # initializing with zero this will only be checked for previously active trades 
        self.sl = 0.0
        self.tp =  0.0 
        self.previousTrade = False






class Signal:
        def __init__(self, timeframe,contract,strategy,side,entryprice,sl,tp):
            self.timeframe :str = timeframe 
            self.contract :Contract = contract

            self.strategy :str = strategy
            self.side :str = side
            
            self.entry_price :float= entryprice 
            self.sl :float = sl
            self.tp :float = tp          
            
            self.key = f"{self.contract.symbol}_{self.timeframe}"

            self.id = f"{self.key} {self.strategy}_{datetime.datetime.now()}"
