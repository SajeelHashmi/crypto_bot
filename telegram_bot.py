import telebot
import os
import threading
import json
import requests
import logging

logger = logging.getLogger()

class Telegram_Bot:

    def __init__(self,disabled = False) -> None:
        self.disabled  =disabled
        self.TOKEN = '7006309540:AAECIlDKImoifCUMQczPb-deHznwhIHX1zg'
        self.bot = telebot.TeleBot(self.TOKEN, parse_mode=None)
        self.my_id = '5731301395'

        self.trades = {}

        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            self.bot.reply_to(message, "Howdy, how are you doing?")

        # @self.bot.poll_answer_handler()
        # def handle_poll(poll):
        #     headers = {'Content-type': 'application/json'}

        #     for t in self.trades:
        #         if self.trades[t] == poll.poll_id:
        #             # send message back to main server

        #             if poll.option_ids[0] == 1:
        #                 print("No Trade")

        #             elif poll.option_ids[0] == 0:
        #                 print("Trade")
        #                 print(t)
        #                 # res = requests.post(
        #                 #     'http://127.0.0.1:5000/trade_telegram',
        #                 #     json={
        #                 #         'id': t,
        #                 #         'trade': 1
        #                 #     },
        #                 #     headers=headers)
        #                 # print(res)

        # # t = threading.Thread(target=self.bot.infinity_polling)
        # # t.start()
        self.send_message("telegram messages initialized working")

    def send_message(self, message):
        if self.disabled:
            logger.info("telegram msg disabled")
            return
        self.bot.send_message(self.my_id, message)
        # self.send_poll(message, tradeId)

    # def send_poll(self, message, trade_id):
    #     answer_options = [
    #         telebot.types.InputPollOption("YES"),
    #         telebot.types.InputPollOption("NO")
    #     ]

    #     poll = self.bot.send_poll(
    #         chat_id=self.my_id,
    #         question=message,
    #         options=answer_options,
    #         type="regular",
    #         allows_multiple_answers=False,
    #         is_anonymous=False,
    #     )
    #     print(poll.json['poll']['id'])
    #     self.trades[trade_id] = poll.json['poll']['id']
