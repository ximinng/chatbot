# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
from chatterbot import ChatBot
from chatterbot.response_selection import get_most_frequent_response
from chatterbot.trainers import ChatterBotCorpusTrainer

'''
This is an example showing how to create an export file from
an existing chat bot that can then be used to train other bots.
'''

bot = ChatBot(name="Andriod",
              # 存储适配器
              storage_adapter='chatterbot.storage.SQLStorageAdapter',
              # 回复适配器 'Math & Time Bot'
              logic_adapters=[
                  {
                      "import_path": "chatterbot.logic.BestMatch",
                      # "statement_comparison_function": chatterbot.comparisons.levenshtein_distance,
                      # "response_selection_method": chatterbot.response_selection.get_first_response,
                      # 'default_response': 'I am sorry, but I do not understand.',
                      # 'maximum_similarity_threshold': 0.10
                  },
                  {
                      'import_path': 'chatterbot.logic.SpecificResponseAdapter',
                      'input_text': 'Help me!',
                      'output_text': 'Ok, here is a link: http://chatterbot.rtfd.org'
                  }
              ],
              # response_selection_method=get_most_frequent_response,
              preprocessors=[
                  'chatterbot.preprocessors.clean_whitespace',
                  'chatterbot.preprocessors.unescape_html',
                  'chatterbot.preprocessors.convert_to_ascii'
              ],
              database_uri='sqlite:///database.sqlite3'
              )

# First, lets train our bot with some data
trainer = ChatterBotCorpusTrainer(bot)

trainer.train('chatterbot.corpus.chinese')

# "chatterbot.corpus.english.greetings",
# "chatterbot.corpus.english.conversations"

# Now we can export the data to a file
trainer.export_for_training('./my_export.json')

# 开始对话
print('开始对话：')
while True:
    try:
        bot_input = bot.get_response(input(">"))
        print(bot_input)

    except(KeyboardInterrupt, EOFError, SystemExit):
        break
