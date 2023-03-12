import jieba.analyse
text = '教学思路清晰，符合“三维目标”要求，教学实施较好，能基本体现“合作、探究、互动、评价”的教学模式，充分体现以学生为主体，促进学生主动学习，课堂容量不大，可增加教学容量，以训练接纳信息和处理信息能力'
res = jieba.analyse.extract_tags(text, withWeight=True, topK=5)
print(res)
for word, weight in res:
    print('%s %s' % (word, weight))



# import pkuseg
#
# seg = pkuseg.pkuseg(model_name='pku')
# text = seg.cut('教学思路清晰，符合“三维目标”要求，教学实施较好，能基本体现“合作、探究、互动、评价”的教学模式，充分体现以学生为主体，促进学生主动学习，课堂容量不大，可增加教学容量，以训练接纳信息和处理信息能力')
# print(text)

# trainFile = "./Data/icwb2-data-master/training/pku_training.utf8"  # 训练文件路径
# # trainFile = "E:\Graduate\Codes\Text-Classification\Data\icwb2-data-master\training\pku_training.utf8"  # 训练文件路径
# testFile = "./Data/icwb2-data-master/testing/pku_test.utf8"  # 测试文件路径
# # testFile = "E:\Graduate\Codes\Text-Classification\Data\icwb2-data-master\testing\pku_test.utf8"  # 测试文件路径
# savedir = "./models/pku"  # 训练模型的保存路径
# train_iter = 20  # 训练轮数
# init_model = './models/default_v2'  # 初始化模型，默认为None表示使用默认初始化，用户可以填自己想要初始化的模型的路径如init_model='./models/'
# # 训练文件为'msr_training.utf8'，测试文件为'msr_test_gold.utf8'，模型存到'./models'目录下，开20个进程训练模型
# # user_dict       设置用户词典。默认为'safe_lexicon'表示我们提供的一个中文词典(仅pip)。用户可以传入一个包含若干自定义单词的迭代器。
# pkuseg.train(trainFile, testFile, savedir, train_iter, init_model)
