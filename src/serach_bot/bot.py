# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
from chatterbot import ChatBot
from chatterbot.response_selection import get_most_frequent_response
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

"""Create a new chat bot"""
bot = ChatBot(name="Andriod")

# 指定语料库
trainer = ChatterBotCorpusTrainer(bot)
trainer.train(
    "chatterbot.corpus.chinese"
)

"""Get a response"""
# response = bot.get_response("Good morning!")
# print(response)

print('Type something to begin...')
while True:
    try:
        bot_input = bot.get_response(input())
        print(bot_input)

    except(KeyboardInterrupt, EOFError, SystemExit):
        break
