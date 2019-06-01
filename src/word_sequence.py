# -*- coding: utf-8 -*-
"""
   Description : 句子编码化处理
   Author :        xxm
"""
import numpy as np


class WordSequence(object):
    PAD_TAG = '<pad>'  # 补位
    UNK_TAG = '<unk>'  # 未知
    START_TAG = '<s>'  # 开始
    END_TAG = '</s>'  # 结束

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        # 初始化编码的字典
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        self.fited = False  # 训练标记

    def to_index(self, word):
        """
        word to index
        encoder
        :return: 如何word在字典中则对应编码，否则返回unknown的编码
        """
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self, index):
        """
        index to word
        :return: decoder
        """
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    def size(self):
        """
        :return: 字典长度
        """
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        return len(self.dict) + 1  # 补位+1

    def __len__(self):
        """
        :return: 系统字典大小
        """
        return self.size()

    def fit(self, sentences, min_count=5, max_count=None, max_features=None):
        """
        定义训练字典方法
        :param sentences: 文本
        :param min_count: 所有文本里字符出现的最小频数限制
        :param max_count: 所有文本里字符出现的最大数限制(不限)
        :param max_features: 文本中字符最大出现次数 --- 特征数
        """

        assert not self.fited, 'WordSequence 只能 fit 一次'

        count = {}  # 词频统计

        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        # 如果大于最小频数，小于最大频数 -> 统计k,v
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}
        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }

        # int型
        if isinstance(max_features, int):
            # 依词频逆序排序
            count = sorted(
                list(count.items()),
                key=lambda x: x[1]  # 排序规则: x[1]==value
            )
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        # 非int型
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)

        self.fited = True

    def transform(self, sentences, max_len=None):
        """
        将句子转化成向量
        :param sentences: 文本
        :param max_len: 最大的长度
        :return: 文本 -> 向量
        """
        assert self.fited, "WordSequence 尚未进行 fit 操作"

        if max_len is not None:
            res = [self.PAD] * max_len
            # print(res)
        else:
            res = [self.PAD] * len(sentences)

        # 遍历所有文本
        for index, line in enumerate(sentences):
            if max_len is not None and index >= len(res):
                break
            # 每一句 to_index
            res[index] = self.to_index(line)

        return np.array(res)

    def inverse_transform(self, vectors,
                          ignore_pad=False, ignore_unk=False,
                          ignore_start=False, ignore_end=False):
        """
        向量转换成句子
        :param vectors: 向量组
        :param ignore_pad ignore_unk ignore_start ignore_end:是否忽略特殊填充位
        :return: 句子
        """
        ret = []
        for vec in vectors:
            word = self.to_word(vec)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)

        return ret


def test():
    ws = WordSequence()
    ws.fit([
        ['我', '的', '家', '在', '西', '安'],
        ['你', '的', '家', '在', '哪'],
        ['不', '知', '道'],
        ['好吧'],
    ])

    vec = ws.transform(['不', '好', '吧'])
    print(vec)

    word = ws.inverse_transform(vec)
    print(word)


if __name__ == '__main__':
    test()
