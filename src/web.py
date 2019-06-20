# -*- coding: utf-8 -*-
"""
   Description :   web interface
   Author :        xxm
"""
import sys
import pickle

import numpy as np
import tensorflow as tf
from flask import Flask, request


def test(params, infos):
    from src.seq_to_seq import SequenceToSequence
    from src.data_utils import batch_flow

    """xiaohaungji语料"""
    x_data, _ = pickle.load(open('data/xiaohaungji_chatbot.pkl', 'rb'))
    ws = pickle.load(open('data/xiaohuangji_ws.pkl', 'rb'))

    # x_data, _ = pickle.load(open('data/chatbot.pkl', 'rb'))
    # ws = pickle.load(open('data/ws.pkl', 'rb'))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './xiaohaungji_model/s2ss_chatbot_anti.ckpt'
    # save_path = './model/s2ss_chatbot_anti.ckpt'

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            x_test = [list(infos.lower())]
            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(pred)

            for p in pred:
                ans = ws.inverse_transform(p)
                print(ans)
                return ans


app = Flask(__name__)


@app.route('/api/chatbot', methods=['get'])
def chatbot():
    infos = request.args['infos']

    import json
    # text = test(json.load(open('params.json')), infos)
    """xiaohaungji语料"""
    text = test(json.load(open('xiaohaungji_model/params.json')), infos)

    # return text
    return "".join(text)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
