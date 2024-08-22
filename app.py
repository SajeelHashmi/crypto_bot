from binance_futures import BinanceFuturesClient
import logging
from models import *
from flask import Flask, request , render_template ,jsonify,redirect,session
import time
import threading
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import os

load_dotenv()  
# from models_binance import Models_binance_client


# TODO 
# APP Complete for on server testing and delivery
# Incorporate Telegram_Bot Class instead of building a new server
# Telegram Bot will be responsible for sending polls and messages
# Telegram bot and binance futures will comunicate with each other directly
# Either telegram bot will have an instance of binance client as a member or vice versa 
# Maybe both like in strategies
# If we make this a class based App then both can be a part of this class and communicate via app as a broker  


socketio = SocketIO()

logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = logging.FileHandler('logs/info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)



app = Flask(__name__)
app.secret_key = 'abc'
socketio.init_app(app, cors_allowed_origins="*")



API_KEY = os.environ.get('BINANCE_PUBLIC_KEY')
API_SECRET = os.environ.get('BINANCE_SECRET_KEY')
BINANCE_CLIENT = BinanceFuturesClient(API_KEY, API_SECRET, False)
# BINANCE_CLIENT.start_models('BTCUSDT','1h',5,1,1)
# BINANCE_CLIENT.start_models('BTCUSDT','1d',5,1,1)

signal_thread = False
tradeThread = False




@socketio.on('connect')
def test_connect():
    logger.info('Client Connected')
    if 'auth' not in session:
        session['auth'] = False
        return False
    auth = session['auth']
    global signal_thread
    global tradeThread
    signal_thread = True
    tradeThread = True
    thread = threading.Thread(target=generateSignal , args=(auth,))
    thread.start()
    thread = threading.Thread(target=tradeSignal , args=(auth,))
    thread.start()
    logger.info('Thread Started')


@socketio.on_error_default
def error_handler(e):
    # this is error function of websocket from our flask app to the front end    
    logger.info('An error has occurred: ' + str(e))



@socketio.on('tradeSignal')
def genTradeSignal():
    if 'auth' not in session:
        session['auth'] = False
        return False
    auth = session['auth']
    global tradeThread
    tradeThread = True
    thread = threading.Thread(target=tradeSignal , args=(auth,))
    thread.start()
    logger.info('Thread Started')


@socketio.on('disconnect')
def disconect():
    logger.info('Client Disconnected')
    global signal_thread
    signal_thread = False

def tradeSignal(auth):
    global BINANCE_CLIENT
    global tradeThread
    if auth:

        while tradeThread:

                try:
                    for trade in BINANCE_CLIENT.active_trades:
                        if trade.status == "open" and trade.entry_price is not None:
                            try:
                                currPrice = BINANCE_CLIENT.contracts[trade.contract.symbol].candles['15m'][-1].close
                            except: 
                                currPrice=0
                            data =  {
                                'contract':trade.contract.symbol,
                                'strategy':trade.strategy,
                                'side':trade.side,
                                'entry': trade.entry_price ,
                                'pnl':trade.pnl,
                                
                                'currentPrice':currPrice,
                                'status':trade.status
                            } 
                                        
                            socketio.emit('tradeSignal',data)
                            time.sleep(0.3)
                    time.sleep(0.3)

                except Exception as e:
                    logger.info(f"Error in sending trades data to front end: {e.with_traceback()}")
    
    else:
        logger.info('Not Authenticated asking for ongoing trades')


def generateSignal(auth):
    global BINANCE_CLIENT
    global signal_thread
    logger.info('here in gen signal')
    if auth:
        while signal_thread:
            # if BINANCE_CLIENT.ws_thread:
                try:
                    for key, row in BINANCE_CLIENT.activeContracts.items():
                        row['trend'] = row['strategies'][0]['strategy'].find_trend()
                        websocketData = {'key' : key, 'trend': row['trend']}
                        websocketData['signals'] =[]
                        for strat in row['strategies']:
                                websocketData['signals'].append(strat['signal'])
                        socketio.emit('signal', websocketData)
                
                    time.sleep(5)
                except Exception as e:
                    continue

    else:
        logger.info('Not Authenticated asking for signals')



@app.route('/',methods=['GET'])
def index():
    global BINANCE_CLIENT
    if 'auth' not in session:
        session['auth'] = False
    contracts = []
    activeCoins = []
    startegies = []
    if session['auth']:
        contracts = [c.symbol for c in BINANCE_CLIENT.contracts.values()]
        startegies = BINANCE_CLIENT.stratNames


    timeframes= [ '1m','3m',"15m","30m","1h", "4h",'1d','3d','1w','1M']
    return render_template('index.html', 
            contracts =contracts,
            authenticated = session['auth'],
            timeframes = timeframes,
            startegies = startegies
            )



# send the startStrategy request via js and return a response with an id for the coin and timeframe
@app.route('/startstrategy',methods=['POST'])
def startStrategy():
    global BINANCE_CLIENT
    if session['auth']:
        if not BINANCE_CLIENT.ws_thread:
            return jsonify(f'Wait a couple mminutes trying to reconnect ws stream')

        data = request.json
        if 'coin' in data and 'timeframe' in data:
            coin = data['coin']
            timeframe = data['timeframe']
            if coin is None or timeframe is None:
                return jsonify({'msg':'Coin and Timeframe are required'})
            symbol = coin.upper()
            try:
                contract = BINANCE_CLIENT.contracts[symbol]
            except KeyError:

                return jsonify(f'Contract {symbol} not found')
            if timeframe not in [ '1m','3m',"15m","30m","1h", "4h",'1d','3d','1w','1M']:
                return jsonify({'msg':f'Timeframe {timeframe} not found'})  

            # remove hardcoded parameters before finalization
            t = threading.Thread(target=BINANCE_CLIENT.startStrategies, args=(symbol,timeframe,0.1,0.1,False))
            t.start()
    
            

            return jsonify({"key":symbol + '_' + timeframe})
    return jsonify ({'msg':'Not Authenticated'})


@app.route('/stopstrategy',methods=['POST'])
def stopStrategy():
    global BINANCE_CLIENT
    if session['auth']:
        data = request.json
        if 'key' in data:
            key = data['key']
            if key == None :
                return jsonify({'msg':'Key is required to turn off strategy'})
            msg  = BINANCE_CLIENT.stop_strategy(key)
            return jsonify({'status':msg})
          
    return jsonify ({'msg':'Not Authenticated'})



@app.route('/authenticate',methods=['POST'])
def authenticate():
    global API_KEY
    global API_SECRET
    api_key = request.form.get('api_key')
    api_secret = request.form.get('api_secret')
    
    if api_key != None  and api_secret  != None:
        if api_key == API_KEY and api_secret == API_SECRET:
            session['auth'] = True
            
    return redirect('/')






@app.route('/logout',methods=['POST'])
def logout():
    global signal_thread
    if 'auth' in session:
        session['auth'] = False
    else:
        session['auth'] = False
    socketio.emit('dissconect', {'msg':'Logged Out'})
    signal_thread = False
    return jsonify({'msg':'Logged Out'})


@app.route('/closeposition',methods=['POST'])
def closePosition():
    logger.info('in close positon')

    global BINANCE_CLIENT
    if session['auth']:
        data = request.json
        if 'symbol' in data and 'strategy' in data:
            symbol = data['symbol']
            strat_name = data['strategy']
            if symbol == None or strat_name ==None  :
                return jsonify({'msg':'Invalid Params'})
            msg  = BINANCE_CLIENT.close_position(symbol,strat_name)
        
            return jsonify({'status':msg})
        else:
            return jsonify({'msg':'Invalid Params'})

    else:        
        return jsonify ({'msg':'Not Authenticated'})



@app.route('/startTest',methods=['GET'])
def startTest():
    global BINANCE_CLIENT
    if session['auth']:
        if not BINANCE_CLIENT.ws_thread:
            return jsonify(f'Wait a couple mminutes trying to reconnect ws stream')
        t = threading.Thread(target=BINANCE_CLIENT.start_test_strats )
        t.start()
        return jsonify({'msg':'success'})

    return jsonify ({'msg':'Not Authenticated'})

@app.route('/stopTest',methods=['GET'])
def stopTest():
    global BINANCE_CLIENT
    if session['auth']:
        if not BINANCE_CLIENT.ws_thread:
            return jsonify(f'Wait a couple mminutes trying to reconnect ws stream')
        t = threading.Thread(target=BINANCE_CLIENT.stop_test_strats )
        t.start()
        return jsonify({'msg':'success'})

    return jsonify ({'msg':'Not Authenticated'})



@app.route('/trade_telegram',methods=['POST'])
def trade_telegram():
    try:
        logger.info("TRADE TELEGRAM ENDPOINT CALLED")
        global BINANCE_CLIENT
        data = request.json
        id = data['id']
        trade = data['trade']
        key = id.split(' ')[0]
        stratName = id.split(' ')[1].split('_')[0]

        
        row = BINANCE_CLIENT.activeContracts[key]
        for strat in row['strategies']:
            if strat['strategy'].stat_name == stratName:
                strat['strategy'].open_position_telegram(id,trade)
                break
        return "SUCCESS"

    except Exception as e:
        return f"exception {e}"
    




class App:
    def __init__(self) -> None:
        
        

        
        self.app = Flask(__name__)
        self.app.secret_key = 'abc'

        self.socketio = SocketIO()
        self.socketio.init_app(self.app, cors_allowed_origins="*")


        self.binance_client = BinanceFuturesClient(API_KEY, API_SECRET, False)
        
        self.signal_thread = False
        self.tradeThread = False




        @self.socketio.on('connect')
        def test_connect():
            print('Client Connected')
            if 'auth' not in session:
                session['auth'] = False
                return False
            auth = session['auth']
            self.signal_thread = True
            self.tradeThread = True
            thread = threading.Thread(target=self.generateSignal , args=(auth,))
            thread.start()
            thread = threading.Thread(target=self.tradeSignal , args=(auth,))
            thread.start()
            print('Thread Started')


        @self.socketio.on_error_default
        def error_handler(e):
            print('An error has occurred: ' + str(e))
    



        @self.socketio.on('tradeSignal')
        def genTradeSignal():
            print('trade signal requested')
            if 'auth' not in session:
                session['auth'] = False
                return False
            auth = session['auth']
            self.tradeThread = True
            thread = threading.Thread(target=self.tradeSignal , args=(auth,))
            thread.start()
            print('Thread Started')


        @self.socketio.on('disconnect')
        def disconect():
            print('Client Disconnected')
            self.signal_thread = False


        @self.app.route('/',methods=['GET'])
        def index():
            print("CLASS BASED APP WORKING")
            if 'auth' not in session:
                session['auth'] = False
            contracts = []
            activeCoins = []
            startegies = []
            if session['auth']:
                contracts = [c.symbol for c in self.binance_client.contracts.values()]
                startegies = self.binance_client.stratNames
                

            timeframes= [ '1m','3m',"15m","30m","1h", "4h",'1d','3d','1w','1M']
            return render_template('index.html', 
                    contracts =contracts,
                    authenticated = session['auth'],
                    timeframes = timeframes,
                    startegies = startegies
                    )



        @self.app.route('/startstrategy',methods=['POST'])
        def startStrategy():
            if session['auth']:
                if not self.binance_client.ws_thread:
                    return jsonify(f'Wait a couple mminutes trying to reconnect ws stream')

                data = request.json
                if 'coin' in data and 'timeframe' in data:
                    coin = data['coin']
                    timeframe = data['timeframe']
                    print(coin, timeframe)
                    if coin == None or timeframe == None:
                        return jsonify({'msg':'Coin and Timeframe are required'})
                    symbol = coin.upper()
                    try:
                        contract = self.binance_client.contracts[symbol]
                    except KeyError:

                        return jsonify(f'Contract {symbol} not found')
                    if timeframe not in [ '1m','3m',"15m","30m","1h", "4h",'1d','3d','1w','1M']:
                        return jsonify({'msg':f'Timeframe {timeframe} not found'})  

                    # remove hardcoded parameters before finalization
                    t = threading.Thread(target=self.binance_client.startStrategies, args=(symbol,timeframe,5,1,False))
                    t.start()
            
                    

                    return jsonify({"key":symbol + '_' + timeframe})
            return jsonify ({'msg':'Not Authenticated'})


        @self.app.route('/stopstrategy',methods=['POST'])
        def stopStrategy():
            print(session['auth'])
            if session['auth']:
                data = request.json
                if 'key' in data:
                    key = data['key']
                    if key == None :
                        return jsonify({'msg':'Key is required to turn off strategy'})
                    msg  = self.binance_client.stop_strategy(key)
                    return jsonify({'status':msg})
                
            return jsonify ({'msg':'Not Authenticated'})



        @self.app.route('/authenticate',methods=['POST'])
        def authenticate():
            api_key = request.form.get('api_key')
            api_secret = request.form.get('api_secret')
            
            if api_key != None  and api_secret  != None:
                if api_key == API_KEY and api_secret == API_SECRET:
                    session['auth'] = True
                    
            print(session['auth'])
            return redirect('/')




        @self.app.route('/logout',methods=['POST'])
        def logout():
            global signal_thread
            if 'auth' in session:
                session['auth'] = False
            else:
                session['auth'] = False
            self.socketio.emit('dissconect', {'msg':'Logged Out'})
            signal_thread = False
            return jsonify({'msg':'Logged Out'})


        @self.app.route('/closeposition',methods=['POST'])
        def closePosition():
            print('in close positon')
            print(session['auth'])

            if session['auth']:
                data = request.json
                if 'symbol' in data and 'strategy' in data:
                    symbol = data['symbol']
                    strat_name = data['strategy']
                    if symbol == None or strat_name ==None  :
                        return jsonify({'msg':'Invalid Params'})
                    msg  = self.binance_client.close_position(symbol,strat_name)
                
                    return jsonify({'status':msg})
                else:
                    print(data)
                    return jsonify({'msg':'Invalid Params'})

            else:        
                return jsonify ({'msg':'Not Authenticated'})



        @self.app.route('/startTest',methods=['GET'])
        def startTest():
            if session['auth']:
                if not self.binance_client.ws_thread:
                    return jsonify(f'Wait a couple mminutes trying to reconnect ws stream')
                t = threading.Thread(target=self.binance_client.start_test_strats )
                t.start()
                return jsonify({'msg':'success'})

            return jsonify ({'msg':'Not Authenticated'})

        @self.app.route('/stopTest',methods=['GET'])
        def stopTest():
            global BINANCE_CLIENT
            if session['auth']:
                if not BINANCE_CLIENT.ws_thread:
                    return jsonify(f'Wait a couple mminutes trying to reconnect ws stream')
                t = threading.Thread(target=BINANCE_CLIENT.stop_test_strats )
                t.start()
                return jsonify({'msg':'success'})

            return jsonify ({'msg':'Not Authenticated'})



        @self.app.route('/trade_telegram',methods=['POST'])
        def trade_telegram():
            try:
                print("TRADE TELEGRAM ENDPOINT CALLED")
                data = request.json
                id = data['id']
                trade = data['trade']
                print(id,trade)
                key = id.split(' ')[0]
                stratName = id.split(' ')[1].split('_')[0]
                print(key,stratName)

                
                row = self.binance_client.activeContracts[key]
                for strat in row['strategies']:
                    if strat['strategy'].stat_name == stratName:
                        strat['strategy'].open_position_telegram(id,trade)
                        break
                return "SUCCESS"

            except Exception as e:
                return f"exception {e}"
            
        self.socketio.run(self.app, debug=False,port=5000)






    def tradeSignal(self,auth):
        if auth:

            while self.tradeThread:
                    try:
                        for trade in self.binance_client.active_trades:
                            if trade.status == "open" and trade.entry_price is not None:
                                try:
                                    currPrice = self.binance_client.prices[trade.contract.symbol]
                                except: 
                                    currPrice=0
                                data =  {
                                    'contract':trade.contract.symbol,
                                    'strategy':trade.strategy,
                                    'side':trade.side,
                                    'entry': trade.entry_price ,
                                    'pnl':trade.pnl,
                                    
                                    'currentPrice':currPrice,
                                    'status':trade.status
                                } 
                                            
                            self.socketio.emit('tradeSignal',data)
                        time.sleep(0.3)

                    except Exception as e:
                        logger.info(f"Error in sending trades data to front end: {e}")
        
        else:
            print('Not Authenticated')


    def generateSignal(self,auth):
        print('here in gen signal')
        if auth:
            while self.signal_thread:
                    try:
                        for key, row in self.binance_client.activeContracts.items():

                            row['trend'] = row['strategies'][0]['strategy'].find_trend()
                            websocketData = {'key' : key, 'trend': row['trend']}
                            websocketData['signals'] =[]
                            
                            for strat in row['strategies']:
                                    websocketData['signals'].append(strat['signal'])
                            
                            self.socketio.emit('signal', websocketData)
                    
                        time.sleep(5)
                    except Exception as e:
                        continue
        else:
            print('Not Authenticated')



if __name__ == "__main__":
    # App()
    socketio.run( app,
                  debug=False,
                  port=5000,
                #   host="0.0.0.0"
                  )




