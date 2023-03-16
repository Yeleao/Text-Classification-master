# -*- coding: utf-8 -*-
import io
import pickle
import sys
from sys import argv

import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences


def callPreList(list_pre):
    # 导入字典
    word_path = 'E:\Graduate\Codes\Text-Classification\Model\word_dict.pk'
    output_path = 'E:\Graduate\Codes\Text-Classification\Model\label_dict.pk'
    model_path = 'E:\Graduate\Codes\Text-Classification\Model\corpus_model.h5'
    with open(word_path, 'rb') as f:
        word_dictionary = pickle.load(f)
        # try:
        #     word_dictionary = pickle.load(f)
        # except KeyboardInterrupt:
        #     print("quit")
        # except Exception as ex:
        #     print("出现如下异常%s" % ex)
    with open(output_path, 'rb') as f:
        output_dictionary = pickle.load(f)
        # try:
        #     output_dictionary = pickle.load(f)
        # except KeyboardInterrupt:
        #     print("quit")
        # except Exception as ex:
        #     print("出现如下异常%s" % ex)
    # 载入模型
    lstm_model = load_model(model_path)
    input_shape = 180
    for i in list_pre:
        try:
            # 数据预处理
            sent = i
            x = [[word_dictionary[word] for word in sent]]
            x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

            # 模型预测
            y_predict = lstm_model.predict(x)
            label_dict = {v: k for k, v in output_dictionary.items()}
            print(sent)
            print(label_dict[np.argmax(y_predict)])

        except KeyError as err:
            print("您输入的句子有汉字不在词汇表中，请重新输入！")
            print("不在词汇表中的单词为：%s." % err)


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    str = argv[1]
    # str = "[教得太好了，所值，真心不错, 这种货色就这样吧，别期待怎样。, 什么教学态度啊，出了事情一个推一个，作业麻烦还有考试, 很满意，教得很好，思路也很清晰。]"
    str = str.strip('[')
    str = str.strip(']')
    str2list = str.split(", ")
    callPreList(str2list)
