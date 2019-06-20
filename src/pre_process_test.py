# -*- coding: utf-8 -*-
"""
   Description : 训练语料库的解压处理
   Author :        xxm
"""
import re
import pickle
import jieba
from tqdm import tqdm


# 去掉非法字符，合并句子
def make_split(line):
    if re.match(r'.*([，…?!\.,!？])$', ''.join(line)):
        return []
    return [', ']


# 句子评判
def good_line(line):
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


# 规则替换
def regular(sen):
    # ...最少连续出现3次 最多100次 替换成…
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'…{2,100}', '…', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    return sen


def main(limit=30,  # 句子长度
         x_limit=1,
         y_limit=2):
    from word_sequence import WordSequence

    print('extract lines')
    """dgk语料"""
    # fp = open("raw_data/dgk_shooter_min.conv", 'r', errors='ignore', encoding='utf-8')
    """xiaohuangji语料"""
    fp = open("raw_data/xiaohaungji50w_test.conv", 'r', errors='ignore', encoding='utf-8')

    # 保存全部句子列表
    groups = []
    # 保存一行
    group = []

    for line in tqdm(fp):  # 显示进度条

        if line.startswith('M '):  # 句子处理M开头
            line = line.replace('\n', '')  # 去掉回撤

            if '/' in line:
                line = line[2:].split('/')  # 去掉斜杠 -> return <list>
                line = list(regular(''.join(line)))  # 去掉词语

                line = jieba.lcut(''.join(line))
            else:
                line = list(line[2:])

            group.append(line)
            # print(group)

        else:  # E开头句子---line.startswith('E ')
            if group:
                groups.append(group)
                group = []

    if group:
        groups.append(group)
        group = []

    print('\nextract group')

    """定义问答对"""
    x_data = []
    y_data = []

    for group in tqdm(groups):
        # print(group)
        for index, line in enumerate(group):
            if index ==0 and good_line(line) : x_data.append(line)
            if index ==1 and good_line(line) : y_data.append(line)

    print(x_data)
    print(y_data)

    # 问答对数据量
    print('\n问句数量：' + str(len(x_data)), '答句数量：' + str(len(y_data)))

    # 将问答对放入zip object(至多20字符)
    for ask, answer in zip(x_data[:30], y_data[:30]):
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 20)
    
    """组装数据"""
    data = list(zip(x_data, y_data))
    
    # 组装规则:
    data = [
        (x, y) for x, y in data
        if len(x) < limit and len(y) < limit and len(y) >= y_limit and len(x) >= x_limit
    ]
    x_data, y_data = zip(*data)
    
    # word_sequence模型训练
    print('fit word_sequence')
    from gensim.models import word2vec
    import gensim
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5


    # ws_input = WordSequence()
    # ws_input.fit(x_data + y_data)
    
    # # 保存 (pickle格式)
    # print('dump')
    # pickle.dump(
    #     (x_data, y_data),
    #     # open('data/dgk_chatbot.pkl', 'wb')
    #     open('data/xiaohaungji_chatbot.pkl', 'wb')
    # )
    # pickle.dump(ws_input, open('data/xiaohuangji_ws.pkl', 'wb'))
    #
    # print('done')


if __name__ == '__main__':
    main()
