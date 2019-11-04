<h1 id="chatbot" align="center">chatbot</h1>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue" alt="Pyhton 3">
    </a>
    <a href="http://www.apache.org/licenses/">
        <img src="https://img.shields.io/badge/license-Apache-blue" alt="GitHub">
    </a>
    <a href="https://github.com/ximingxing/Machine-Learning-in-Action/pulls">
        <img src="https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=flat-square" alt="welcome">
    </a>
</p>

<p align="center">
    <a href="#clipboard-getting-started">快速开始 - Getting Started</a> •
    <a href="#table-of-contents">内容 - Table of Contents</a> •
    <a href="#about">关于 - About</a> •
    <a href="#acknowledgment">鸣谢 - Acknowledgment</a> •
    <a href="#speech_balloon-faq">问题 - FAQ</a> •
</p>

<h6 align="center">Made by ximing Xing • :milky_way: 
<a href="https://ximingxing.github.io/">https://ximingxing.github.io/</a>
</h6>

智能聊天机器人作为自然语言处理的一个重要分支， 是目前最火热也最具挑战的研究方向，它对于促进人机交互方式 的发展有着重要的意义。

本项目可用作学习使用或毕业设计，相关问题可与我联系。

Open Source runs on love, laughter and a whole lot of coffee. 
Consider buying me one if you find this content useful ☕️😉.

<h2 align="center">:clipboard: 快速开始 - Getting Started</h2>

1. 执行`cd chatbot` -- cd to the directory where requirements.txt is located
2. 开启你的虚拟环境（conda env） -- activate your virtualenv
3. 执行`pip install -r requirements.txt in your shell` -- run:  `pip install -r requirements.txt in your shell`

**至此，你已经安装了本项目所需要的全部环境**

4. 执行`cd src/chatbot_quick_start`

**在训练模型前一定要先下载数据,可在`CONFIG.py`中的`"path": "data/xiaohuangji50w_fenciA.conv"`处指定路径**

5. 执行`python Train.py`, 模型训练

    模型超参数保存在`CONFIG.py`

6. 模型训练结束后，运行`RestfulAPI.py`启动web服务

7. 访问`localhost:8000/api/chatbot?infos=你好` 即可看到回复   

<h2 align="center">内容 - Table of Contents</h2>
<p align="right"><a href="#chatbot"><sup>▴ Back to top</sup></a></p>

**如果你想了解`快速开始`中的详细内容，可以阅读这个环节**

1. `extract_conv.py`or`new_extract_conv.py` 解压并预处理语料文件

    * `raw_data/` : 用于存放原始语料(.conv格式)
    
    * `data/` : 预处理后的语料 (.pickle格式)
    
2. `params.json` 模型超参数

3. `word_sequence.py` 对文本分词并编码

4. `seq_to_seq.py` attentive Encoder-Decoder with LSTM

5. `train.py`or`anti_train.py` 训练模型

    * model/ : 存放训练好的模型
    
6. `test.py` 测试模型    

7. `web.py` 提供restful接口的api

<h2 align="center">关于 - About</h2>

* []数据 -- 使用互联网公开的数据集:

    * 中文电影对话 dgk_shooter_min.conv
    
    * 小黄鸡语料 xiaohuangji50w_fenciA.conv
    
**你可以在[这里](https://github.com/candlewill/Dialog_Corpus)找到数据**

* NLP相关    
    
    * Word embedding 词嵌入
    
    词嵌入(Word embedding)又被称为词表示(Word representation)，每个单词套用该模型后可以转换为一个实数，且每个实数对应词典中的一个特定单词。
    它是一种用于在低维的词向量空间中用来学习深层的单词表示的技术，通过对词汇量的扩大，可以很大地提升训练速度，因为会通过在词嵌入空间中非常相近的单词来共享一些信息。
    常用的词嵌入模型有 Word2Vec，该模型是由包含了由一千多亿单词组成的 Google 新闻数据训练的，并且被证明该模型在一个非常广泛的数据集上展现出了强有力的信息。
    
    * Encoder-decoder 加解密模型
    
    ![Encoder-Decoder](https://github.com/learnmedicalcantsavecn/chatbot/blob/master/img/encoder-decoder.png)
    
    在以往的研究中，我们会发现实际上智能对话系统问题可以被很好地应用到的自然语言的机器翻译框架中，我们可以将用户提出的问题作文输入机器翻译模型的源序列，
    系统返回的答案则可以作为翻译模型的目标序列。因此，机器翻译领域相对成熟的技术与问答系统所需要的框架模型有了很好的可比性，Ritter 等人借鉴了统计机器翻译的手段，
    使用 Twitter 上的未被结构化的对话数据集，提出了一个问答生成模型的框架。
    Encoder-decoder 框架目前发展较为成熟，在文本处理领域已经成为一种研究模式，可应用场景十分广泛。
    它除了在已有的文本摘要提取、机器翻译、词句法分析方面有很大的贡献之外，在本课题中，也可以被应用到人机对话和智能问答领域。
    
    * Attention 注意力机制
    
    ![Attention](https://github.com/learnmedicalcantsavecn/chatbot/blob/master/img/attention.png)
    
    Attention 结构的核心优点就是通过在模型“decoder”阶段对相关的源内容给予“关注”，从而可以在目标句子和源句子之间建立直接又简短的连接，解决机器人模型和用户之间的信息断层问题。
    注意力机制如今作为一种事实标准，已经被有效地应用到很多其他的领域中，比如图片捕获生成，语音识别以及文字摘要等。
    在传统 seq2seq 模型的解码过程中，“encoder”加密器的源序列的最后状态会被作为输入，直接传递到“decoder”解码器。
    直接传递固定且单一维度的隐藏状态到解码器的方法，对于简短句或中句会有较为可观的效果，却会成为较长的序列的信息瓶颈。
    然而，不像在 RNN 模型中将计算出来的隐藏层状态全部丢弃，注意力机制为我们提供了一种方法，可以使解码器对于源序列中的信息选择重点后进行动态记忆。
    也就是说，通过注意力机制，长句子的翻译质量也可以得到大幅度的提升。
    

<h2 align="center">Acknowledgment</h2>
<p align="right"><a href="#chatbot"><sup>▴ Back to top</sup></a></p>

感谢这些Paper给了我启示：

[智能聊天机器人的技术综述](https://github.com/ximingxing/chatbot/blob/master/paper/%E6%99%BA%E8%83%BD%E8%81%8A%E5%A4%A9%E6%9C%BA%E5%99%A8%E4%BA%BA%E7%9A%84%E6%8A%80%E6%9C%AF%E7%BB%BC%E8%BF%B0.pdf)

[AliMe Chat A Sequence to Sequence and Rerank based Chatbot Engine](https://github.com/ximingxing/chatbot/blob/master/)

[Neural Responding Machine for Short-Text Conversation]()

[Sequence to Sequence Learningwith Neural Networks]()

Search for a specific pattern. Can't find one? Please report a new pattern [here](https://github.com/ximingxing/Machine-Learning-in-Action/issues).

<h2 align="center">:speech_balloon: FAQ</h2>
<p align="right"><a href="#chatbot"><sup>▴ Back to top</sup></a></p>
