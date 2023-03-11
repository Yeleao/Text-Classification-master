# coding:utf-8
import pickle
import numpy as np
import keras


class CNN_call():
    def callPredict(self, str):
        # 导出保存的模型
        with open('dump/corpus/clf_preprocess.pkl', 'rb') as clf_preprocess:
            clf = pickle.load(clf_preprocess)

        model = keras.models.load_model('dump/corpus/clf_model.h5')
        clf.model = model

        x_test = [str]
        predict_rate = clf.predict(x_test)
        max_rate = np.max(predict_rate)

        for i in predict_rate:
            # 取出i中元素最大值所对应的索引
            predict_lable = clf.preprocess.label_set[i.argmax()]
        return predict_lable, max_rate

    callPredict()
