from paddlenlp import Taskflow
summarizer = Taskflow("text_summarization")

text = '63岁退休教师谢淑华，拉着人力板车，历时1年，走了2万4千里路，带着年过九旬的妈妈环游中国，完成了妈妈“一辈子在锅台边转，也想出去走走”的心愿。她说：“妈妈愿意出去走走，我就愿意拉着，孝心不能等，能走多远就走多远。'
print(f'Content: {text}\n')
title = summarizer(text)
print(f'Title: {title[0]}')

