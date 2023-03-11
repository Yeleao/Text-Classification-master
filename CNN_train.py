# coding:utf-8
from TextClassification import TextClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import os

import pickle


# 导入数据
def importData(data_type, data_name):
    projectpath = os.path.dirname(os.path.abspath(__file__))

    data = pd.read_csv(projectpath + "/Data/" + data_name, encoding='utf8')
    x = data['text']
    y = [[i] for i in data['label']]

    # 拆分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=1)
    clf = TextClassification()

    # 以下是训练过程
    texts_seq, texts_labels = clf.get_preprocess(x_train, y_train, word_len=1, num_words=2000, sentence_len=50)

    # clf有两个成员变量 model和preprocess  此时还没有生成model 先序列化包含preprocess的clf
    with open('dump/clf_preprocess.pkl', 'wb') as f:
        pickle.dump(clf, f)

    clf.fit(texts_seq=texts_seq, texts_labels=texts_labels, output_type=data_type, epochs=10, batch_size=64, model=None)

    # 此时已经生成model 仅仅序列化model
    clf.model.save('./dump/clf_model.h5')

    print(data_name + " has been trained")


if __name__ == '__main__':
    data_type = 'single'
    data_name = 'ch_auto.csv'
    importData(data_type, data_name)
