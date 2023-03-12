import pkuseg

seg = pkuseg.pkuseg(model_name='default_v2')
text = seg.cut('吃了3天的头孢和阿奇，没有好转，医生开了雾化治疗的处方单')
print(text)

trainFile = ""  # 训练文件路径
testFile = ""  # 测试文件路径
savedir = "./models/ready"  # 训练模型的保存路径
train_iter = 20  # 训练轮数
init_model = './models/default_v2'  # 初始化模型，默认为None表示使用默认初始化，用户可以填自己想要初始化的模型的路径如init_model='./models/'
# 训练文件为'msr_training.utf8'，测试文件为'msr_test_gold.utf8'，模型存到'./models'目录下，开20个进程训练模型
# user_dict       设置用户词典。默认为'safe_lexicon'表示我们提供的一个中文词典(仅pip)。用户可以传入一个包含若干自定义单词的迭代器。
pkuseg.train(trainFile, testFile, savedir, train_iter, init_model)
