# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
from chatterbot import ChatBot, chatterbot
from chatterbot.response_selection import get_most_frequent_response
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

"""Create a new chat bot"""
bot = ChatBot(name="Andriod",
              # 存储适配器
              storage_adapter='chatterbot.storage.SQLStorageAdapter',
              # 回复适配器 'Math & Time Bot'
              logic_adapters=[
                  {
                      "import_path": "chatterbot.logic.BestMatch",
                      # "statement_comparison_function": chatterbot.comparisons.levenshtein_distance,
                      # "response_selection_method": chatterbot.response_selection.get_first_response,
                      'default_response': 'I am sorry, but I do not understand.',
                      'maximum_similarity_threshold': 0.90
                  },
                  {
                      'import_path': 'chatterbot.logic.SpecificResponseAdapter',
                      'input_text': 'Help me!',
                      'output_text': 'Ok, here is a link: http://chatterbot.rtfd.org'
                  }
              ],
              response_selection_method=get_most_frequent_response,
              preprocessors=[
                  'chatterbot.preprocessors.clean_whitespace',
                  'chatterbot.preprocessors.unescape_html',
                  'chatterbot.preprocessors.convert_to_ascii'
              ],
              database_uri='sqlite:///database.sqlite3'
              )

"""Training your ChatBot"""
conversation = [
    "Hello",
    "Hi there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome."
]

# trainer = ListTrainer(bot)
# trainer.train(conversation)

# 指定语料库
trainer = ChatterBotCorpusTrainer(bot)
trainer.train(
    "chatterbot.corpus.english"
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
