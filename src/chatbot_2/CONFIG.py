# encoding: utf-8

# 模型保存目录
BASE_MODEL_DIR = 'model'

# 模型名称
MODEL_NAME = 'chatbot_model.ckpt'

# 训练轮数
n_epoch = 200
# batch样本数
batch_size = 256
# 训练时dropout的保留比例
keep_prob = 0.8

# 有关语料数据的配置
data_config = {
    # 问题最短的长度
    "min_q_len": 1,
    # 问题最长的长度
    "max_q_len": 20,
    # 答案最短的长度
    "min_a_len": 2,
    # 答案最长的长度
    "max_a_len": 20,
    # 词与索引对应的文件
    "word2index_path": "data/w2i.pkl",
    # 原始语料路径
    "path": "data/xiaohuangji50w_fenciA.conv",
    # 原始语料经过预处理之后的保存路径
    "processed_path": "data/data.pkl",
}

# 有关模型相关参数的配置
model_config = {
    # rnn神经元单元的状态数
    "hidden_size": 256,
    # rnn神经元单元类型，可以为lstm或gru
    "cell_type": "lstm",
    # 编码器和解码器的层数
    "layer_size": 4,
    # 词嵌入的维度
    "embedding_dim": 300,
    # 编码器和解码器是否共用词嵌入
    "share_embedding": True,
    # 解码允许的最大步数
    "max_decode_step": 80,
    # 梯度裁剪的阈值
    "max_gradient_norm": 3.0,
    # 学习率初始值
    "learning_rate": 0.001,
    "decay_step": 100000,
    # 学习率允许的最小值
    "min_learning_rate":1e-6,
    # 编码器是否使用双向rnn
    "bidirection":True,
    # BeamSearch时的宽度
    "beam_width":200
}