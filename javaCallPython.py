# -*- coding: utf-8 -*-
import io
import pickle
import sys
from sys import argv

import keras
import numpy as np


def callPre(str):
    # 导出保存的模型
    pkl_path = 'E:\Graduate\Codes\Text-Classification-master\dump\clf_preprocess.pkl'
    model_path = 'E:\Graduate\Codes\Text-Classification-master\dump\clf_model.h5'

    with open(pkl_path, 'rb') as clf_preprocess:
        try:
            clf = pickle.load(clf_preprocess)
        except KeyboardInterrupt:
            print("quit")
        except Exception as ex:
            print("出现如下异常%s" % ex)
    model = keras.models.load_model(model_path)
    clf.model = model
    x_test = [str]
    predict_rate = clf.predict(x_test)
    max_rate = np.max(predict_rate)

    for i in predict_rate:
        # 取出i中元素最大值所对应的索引
        predict_lable = clf.preprocess.label_set[i.argmax()]
    print(predict_lable)
    # print(max_rate)


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    str = argv[1]
    # str = "[我觉得不行，太垃圾了，建议别上, 我觉得可以，不过建议再多准备, 有些无聊，建议多讲些故事, 挺有趣的，就是知识太少了，没什么用]"
    str = str.strip('[')
    str = str.strip(']')
    str2list = str.split(", ")
    for i in str2list:
        print(i)
        callPre(i)
