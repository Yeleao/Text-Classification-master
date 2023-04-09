# -*- coding: utf-8 -*-
import io
import sys
from sys import argv
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def T5(text):
    # 分词器采样的是T5的分词器
    tokenizer = AutoTokenizer.from_pretrained('./t5-v1_1-small-chinese-cluecorpussmall')
    model = AutoModelForSeq2SeqLM.from_pretrained('./chisum2')
    sentence = text
    input = tokenizer(sentence, max_length=128, truncation=True, return_tensors='pt')  # 对句子进行编码
    del input['token_type_ids']
    output = model.generate(
        **input,
        do_sample=True,  # 是否抽样
        num_beams=3,  # beam search
        bad_words_ids=[[101], [100]],  # bad_word_ids表示这些词不会出现在结果中
        # length_penalty=100,   # 长度的惩罚项
        max_length=100,  # 输出的最大长度
        repetition_penalty=5.0  # 重复的惩罚
    )
    summary = tokenizer.decode(output[0]).split('[SEP]')[0].replace('[CLS]', '').replace(' ', '')
    # print(summary)
    return summary


if __name__ == '__main__':
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # str = argv[1]
    str = "每经AI快讯，2月4日，长江证券研究所金属行业首席分析师王鹤涛表示，2023年海外经济衰退，美债现处于历史高位，黄金的趋势是值得关注的；在国内需求修复的过程中，看好大金属品种中的铜铝钢。此外，在细分的小品种里，建议关注两条主线，一是新能源，比如锂、钴、镍、稀土，二是专精特新主线。（央视财经）"
    # print(str)
    print(T5(str))
