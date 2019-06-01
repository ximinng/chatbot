# -*- coding: utf-8 -*-
"""
   Description :   数据测试训练
   Author :        xxm
"""
import sys
import random
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def test(params):
    from src.seq_to_seq import SequenceToSequence
    from src.data_utils import batch_flow_bucket as batch_flow
    from src.thread_generator import ThreadedGenerator

    """dgk语料"""
    # x_data, y_data = pickle.load(open('chatbot.pkl', 'rb'))
    # ws = pickle.load(open('ws.pkl', 'rb'))
    """xiaohaungji语料"""
    x_data, y_data = pickle.load(open('data/xiaohaungji_chatbot.pkl', 'rb'))
    ws = pickle.load(open('data/xiaohuangji_ws.pkl', 'rb'))

    """训练"""
    n_epoch = 1  # 训练轮数
    batch_size = 128

    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        allow_soft_placement=True,  # CPU/GPU切换
        log_device_placement=False  # 日志
    )

    # 模型保存的路径
    # save_path = './model/s2ss_chatbot.ckpt'
    save_path = './xiaohaungji_model/s2ss_chatbot_anti.ckpt'

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:
            # 定义模型
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )

            # 初始化参数
            init = tf.global_variables_initializer()
            sess.run(init)

            flow = ThreadedGenerator(
                batch_flow([x_data, y_data], ws, batch_size, add_end=[False, True]),
                queue_maxsize=30
            )

            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps),
                           total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    # [[1,2], [3,4]]
                    # [[3,4], [1,2]]
                    x = np.flip(x, axis=1)
                    cost, lr = model.train(sess, x, xl, y, yl, return_lr=True)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(epoch, np.mean(costs), lr))

                model.save(sess, save_path)

    # 测试
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=12,
        parallel_iterations=1,
        **params
    )

    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break


def main():
    import json
    """dgk语料"""
    # test(json.load(open('params.json')))
    """xiaohaungji语料"""
    test(json.load(open('xiaohaungji_model/params.json')))


if __name__ == '__main__':
    main()
