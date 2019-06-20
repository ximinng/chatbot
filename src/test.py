# -*- coding: utf-8 -*-
"""
   Description :   测试
   Author :        xxm
"""
import sys
import pickle

import numpy as np
import tensorflow as tf


def test(params):
    from seq_to_seq import SequenceToSequence
    from data_utils import batch_flow

    """dgk语料"""
    # x_data, _ = pickle.load(open('chatbot.pkl', 'rb'))
    # ws = pickle.load(open('ws.pkl', 'rb'))

    """xiaohaungji语料"""
    # x_data, _ = pickle.load(open('data/xiaohaungji_chatbot.pkl', 'rb'))
    # ws = pickle.load(open('data/xiaohuangji_ws.pkl', 'rb'))

    # 取前五条数据
    # for x in x_data[:5]:
    #     print(' '.join(x))

    # GPU or CPU
    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    # 读取模型路径
    # save_path = './xiaohaungji_model/s2ss_chatbot_anti.ckpt'
    # save_path = './model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        # batch_size=256,
        mode='decode',
        # beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            user_text = input('请输入您的句子:')
            if user_text in ('exit', 'quit'):
                exit(0)
            x_test = [list(user_text.lower())]
            bar = batch_flow(data=[x_test], ws=ws, batch_size=1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            print(x, xl)

            pred = model_pred.predict(
                sess,
                encoder_inputs=np.array(x),
                encoder_inputs_length=np.array(xl)
            )
            print(pred)

            print(ws.inverse_transform(x[0]))

            for p in pred:
                ans = ws.inverse_transform(p)
                print(ans)
                return ans


def main():
    import json

    """dgk语料"""
    # test(json.load(open('params.json')))
    """xiaohaungji语料"""
    # test(json.load(open('xiaohaungji_model/params.json')))


if __name__ == '__main__':
    main()
