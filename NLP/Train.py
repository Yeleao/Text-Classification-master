# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 导入数据
# 文件的数据中，特征为evaluation, 类别为label.
def load_data(filepath_load, input_shape_load=20):
    df = pd.read_csv(filepath_load)

    # 标签及词汇表
    labels, vocabulary = list(df['label'].unique()), list(df['text'].unique())

    # 构造字符级别的特征
    string = ''
    for word in vocabulary:
        string += word

    vocabulary = set(string)

    # 字典列表
    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
    with open('../Model/word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('../Model/label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    vocab_size = len(word_dictionary.keys())  # 词汇表大小
    label_size = len(label_dictionary.keys())  # 标签类别数量

    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df['text']]
    x = pad_sequences(maxlen=input_shape_load, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['label']]
    y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary

# 创建深度学习模型， Embedding + LSTM + Softmax.
def create_LSTM(n_units, input_shape_lstm, output_dim, filepath_lstm):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath_lstm)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape_lstm, mask_zero=True))
    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='../picture/model_lstm.png', show_shapes=True)
    model.summary()

    return model

# 模型训练
def model_train(input_shape_train, filepath_train, model_save_path_train):

    # 将数据集分为训练集和测试集，占比为9:1
    # input_shape = 100
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath_train, input_shape_train)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)

    # 模型输入参数，需要自己根据需要调整
    n_units = 100
    batch_size = 32
    epochs = 5
    output_dim = 20

    # 模型训练
    lstm_model = create_LSTM(n_units, input_shape_train, output_dim, filepath_train)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    lstm_model.save(model_save_path_train)

    n = test_x.shape[0]  # 测试的条数
    predict = []
    label = []
    for start, end in zip(range(0, n, 1), range(1, n+1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        print(''.join(sentence), label_true, label_predict)  # 输出预测结果
        predict.append(label_predict)
        label.append(label_true)

    acc = accuracy_score(predict, label)  # 预测准确率
    print('模型在测试集上的准确率为: %s.' % acc)


if __name__ == '__main__':
    filepath = '../Data/ch_auto.csv'
    input_shape = 180
    model_save_path = '../Model/ch_auto2.h5'
    model_train(input_shape, filepath, model_save_path)
