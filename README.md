# 聊天机器人 chatbot

## DATA

使用互联网公开的数据集:

* 中文电影对话 dgk_shooter_min.conv

* 小黄鸡语料 xiaohuangji50w_fenciA.conv

## NLP Basic Conception

* 词向量(word embedding) : 将来自词汇表的单词或短语映射到实数的向量

* Word2vec : 产生词向量的模型。浅而双层的神经网络，用来训练将词映射到k维的实数向量，并通过向量间的距离来判断词的相似程度

## HOW TO USE

1. extract_conv.py 解压并预处理语料文件

    * raw_data/ : 用于存放原始语料
    
    * data/ : 预处理后的语料 (pickle格式)
    
2. params.json 调整神经网络参数

3. train.py 训练网络

    * model/ : 存放训练好的模型

4. web.py 提供restful接口的api