# -*- coding: utf-8 -*-
import io
import os
import sys
from sys import argv
# Import the necessary modules
import pickle
import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入字典
with open('E:\Graduate\Codes\Text-Classification\Model\word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('E:\Graduate\Codes\Text-Classification\Model\label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)


def callPre(string):
    try:
        # 数据预处理
        input_shape = 180
        sent = string
        x = [[word_dictionary[word] for word in sent]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

        # 载入模型
        model_save_path = 'E:\Graduate\Codes\Text-Classification\Model\ch_auto2.h5'
        lstm_model = load_model(model_save_path)

        # 模型预测
        y_predict = lstm_model.predict(x)
        label_dict = {v: k for k, v in output_dictionary.items()}
        # print('输入语句: %s' % sent)
        print('%s' % label_dict[np.argmax(y_predict)])

    except KeyError as err:
        print("您输入的句子有汉字不在词汇表中，请重新输入！")
        print("不在词汇表中的单词为：%s." % err)


if __name__ == '__main__':
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # str = argv[1]
    str = "这种货色就这样吧，别期待怎样。"
    callPre(str)
