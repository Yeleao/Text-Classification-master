# coding:utf-8
from CNN_call import CNN_call

str = "真好"
predict_lable, max_rate = CNN_call().callPredict(str)

print(str)
print(predict_lable)
print(max_rate)