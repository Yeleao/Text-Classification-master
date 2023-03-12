# -*- coding: utf-8 -*-
import io
import sys
from sys import argv
# Import the necessary modules
import pickle
import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences

# 导入字典
with open('word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)

def callPre(string):

    try:
        # 数据预处理
        input_shape = 180
        sent = string
        x = [[word_dictionary[word] for word in sent]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

        # 载入模型
        model_save_path = '../../Model/corpus_model.h5'
        lstm_model = load_model(model_save_path)

        # 模型预测
        y_predict = lstm_model.predict(x)
        label_dict = {v: k for k, v in output_dictionary.items()}
        print('输入语句: %s' % sent)
        print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])

    except KeyError as err:
        print("您输入的句子有汉字不在词汇表中，请重新输入！")
        print("不在词汇表中的单词为：%s." % err)


if __name__ == '__main__':
    str2 = "[我觉得不行，太垃圾了，建议别上, 我觉得可以，不过建议再多准备, 有些无聊，建议多讲些故事, 挺有趣的，就是知识太少了，没什么用]"
    str1 = "[教得太好了，让人想学习更多知识, 太垃圾了，建议别上, 学到了很多知识，但是有点无聊, 还行吧，比较无聊]"
    str = "[物超所值，真心不错, 很大很好，方便安装！, 这种货色就这样吧，别期待怎样。, 京东服务很好！但我买的这款电视两天后就出现这样的问题，很后悔买了这样的电视]"
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # str = argv[1]
    str = str.strip('[')
    str = str.strip(']')
    str2list = str.split(", ")
    for i in str2list:
        print(i)
        callPre(i)
