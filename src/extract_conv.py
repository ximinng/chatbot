# -*- coding: utf-8 -*-
"""
   Description : 训练语料库的解压处理
   Author :        xxm
"""
import re
import pickle
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


def main(limit=20, x_limit=3, y_limit=6):
    from word_sequence import WordSequence

    # 解压文件
    print('extract lines')
    fp = open("raw_data/dgk_shooter_min.conv", 'r', errors='ignore', encoding='utf-8')
    # 保存全部句子列表
    groups = []
    # 保存一行
    group = []

    # 进度条显示
    for line in tqdm(fp):
        # 句子处理M开头
        if line.startswith('M '):
            # 去掉回撤
            line = line.replace('\n', '')
            # 去掉 '/'
            if '/' in line:
                line = line[2:].split('/')
            else:
                line = list(line[2:])
            line = line[:-1]

            group.append(list(regular(''.join(line))))
        # E开头句子---line.startswith('E ')
        else:
            if group:
                groups.append(group)
                group = []
    if group:
        groups.append(group)
        group = []
    print('extract group')

    # 定义问答对
    x_data = []
    y_data = []

    # 进度条显示
    for group in tqdm(groups):
        # i:行号 , line:内容
        for i, line in enumerate(group):
            last_line = None
            # 至少两行
            if i > 0:
                last_line = group[i - 1]
                if not good_line(last_line):
                    last_line = None
            next_line = None
            if i < len(group) - 1:
                next_line = group[i + 1]
                if not good_line(next_line):
                    next_line = None
            next_next_line = None
            if i < len(group) - 2:
                next_next_line = group[i + 2]
                if not good_line(next_next_line):
                    next_next_line = None

            if next_line:
                x_data.append(line)
                y_data.append(next_line)
            if last_line and next_line:
                x_data.append(last_line + make_split(last_line) + line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) + next_next_line)

    # 问答对数据量
    print(len(x_data), len(y_data))

    # 将问答对放入zip object(至多20字符)
    for ask, answer in zip(x_data[:20], y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 20)

    data = list(zip(x_data, y_data))

    # 组装规则: 3<x<20 and 6<y<20
    data = [
        (x, y)
        for x, y in data
        if len(x) < limit \
           and len(y) < limit \
           and len(y) >= y_limit \
           and len(x) >= x_limit
    ]
    x_data, y_data = zip(*data)

    # word_sequence模型训练
    print('fit word_sequence')
    ws_input = WordSequence()
    ws_input.fit(x_data + y_data)

    # 保存 (bit形式)
    print('dump')
    pickle.dump(
        (x_data, y_data),
        open('chatbot.pkl', 'wb')
    )
    pickle.dump(ws_input, open('ws.pkl', 'wb'))

    print('done')


if __name__ == '__main__':
    main()
