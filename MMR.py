import jieba  #中文分词库
from sklearn.metrics.pairwise import cosine_similarity  #计算两个矩阵之间的相似度
import networkx as nx  #复杂网络的库,这里可直接调用pagerank算法
from sklearn.preprocessing import  MinMaxScaler
import numpy as np
import re  #正则表达式

stopwords_dir = './stopwords.txt'
local_vocab_dir = './local_vocab.txt'
important_dir = './important_people_orgnazation.txt'
content = """内容"""
title = '标题'

# 加载stopwords
with open(stopwords_dir, 'r',encoding='utf8') as f:
    stopwords = set([w.strip() for w in f])

# 加载词向量 local_word_embeddings = {word->id}
local_word_embeddings = {}
with open(local_vocab_dir, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        local_word_embeddings[word] = embedding

# 加载重要词 people_org_set={'易纲', '彭博社',...}
people_org_set = set()
with open(important_dir , 'r', encoding='utf-8') as f:
    for line in f:
        people_org_set.add(line.strip())

# ********** 分句, 计算句子向量 ********** #
# 分句,清理标点符号
def split_document(para):
    # 根据。！；？，、\n 分段
    line_split = re.split(r'[|。|！|；|？|]|\n|，', para.strip())
    # 去除符号
    _seg_sents = [re.sub(r'[^\w\s]','',sent) for sent in line_split]
    return _seg_sents

# 对句子进行分词，得到词库
def _seg_sent(sentence):
    # 去掉非汉字字符 sentence = 新冠肺炎疫情暴发以来
    sentence = re.sub(r'[^\u4e00-\u9fa5]+','',sentence)
    # sentence_depart = 新冠 / 肺炎 / 疫情 / 暴发 / 以来
    sentence_depart = jieba.cut(sentence.strip())
    word_list = []
    for word in sentence_depart:
        if word not in stopwords:
            word_list.append(word)
    return word_list

# 将文章转化成句子向量
# 得到句子对应的向量
def _vector(words):
    # words=['新冠', '肺炎', '疫情', '暴发']->[['新'],[],[],[]]
    words = [w for w in words if w in local_word_embeddings]
    # 求句子平均值
    words_vector = np.mean([local_word_embeddings[w] for w in words], axis=0) \
        if words else np.zeros(300)
    return words_vector
# 得到文章中一个句子，一个向量
def get_sentnce_vector(all_sentences_words):
    # 得到文章中一个句子，一个向量
    sentence_vec = np.array([_vector(words) for words in all_sentences_words])
    return sentence_vec

# 分句,清理标点符号 sentences=['新冠肺炎疫情暴发以来', '频繁出现的无症状感染者病例',...]
sentences = split_document(content)
# 对句子进行分词，得到词库 all_sentences_words=[['新冠', '肺炎', '疫情', '暴发'], ['频繁', '无症状', '感染者', '病例'],...]
all_sentences_words = [_seg_sent(sen) for sen in sentences]
# 将文章转化成句子向量 sentence_vec=[144,300],144个句子，300为词向量大小
sentence_vec = get_sentnce_vector(all_sentences_words)


# ********** 计算textrank ********** #
# 计算textrank，的到每个句子的重要度
def _calc_text_rank(sentence_vec):
    # sentence_vec：[]
    sim_mat = cosine_similarity(sentence_vec) # 只有x，计算x所有sample的笛卡尔积，
    np.fill_diagonal(sim_mat, 0) # 对角线置0
    nx_graph = nx.from_numpy_array(sim_mat)# 转成图
    # max_iter：最大迭代次数，tol：迭代阈值，若两次迭代差值低于该值，则跳出迭代
    tol, max_iter = 1e-7, 1000
    Flag = True
    while Flag:
        try:
            # pagerank算法计算句子的重要度
            pagerank_score = nx.pagerank(nx_graph, tol=tol, max_iter=max_iter)
            Flag = False
        except nx.PowerIterationFailedConvergence as e:
            print(e)
            tol *= 10
    # 只是转成矩阵形式，没有排序
    pagerank_score = np.array([v for k, v in sorted(pagerank_score.items(), key=lambda x: x[0])])
    return pagerank_score

# 计算textrank，的到每个句子的重要度pagerank_score=[0.00846477 0.01014994 0.00592938 ...]
pagerank_score = _calc_text_rank(sentence_vec)


# ********** 计算各特征:1.每个句子是否包含重要词语(时间词,总结词等)********** #
## 是否包含时间这个特征(一般新闻的重要信息都是包含时间节点的)
def have_date(sentence):
    if re.findall('[0-9去上前明后]{1,4}年', sentence):
        return 1
    if re.findall('[0-9上个一二三四五六七八九十]{1,2}月', sentence):
        return 1
    if re.findall('[0-9上昨前]{1,4}日', sentence):
        return 1
    if re.findall('[昨|前]天', sentence):
        return 1
    return 0

# 是否有重要的词汇
def have_important_org_peo(sentence):
    for entity in people_org_set:
        if entity in sentence:
            return 1
    return 0
# 每个句子是否包含重要词语(时间词,总结词等),有1.5,无1
def get_entities_score(sentence):
    date_score = have_date(sentence)
    ple_org_score = have_important_org_peo(sentence)
    return 1.5 if (date_score + ple_org_score) > 0 else 1

# 每个句子是否包含重要词语(时间词,总结词等),有1.5,无1 entities_score=[1.  1.  1.  1.  1.  1.5...]
entities_score = np.array([get_entities_score(sen) for sen in sentences])


# ********** 计算各特征:2.每个句子与标题的关联 ********** #
# 计算标题与所有句子的cos相似度
def get_title_similarity(sentence_vec, title_vec):
    sim_mat = cosine_similarity(sentence_vec,title_vec)
    return sim_mat


## 如果标题与内容句子有3个共有的token，赋值1.5，否则1
def get_title_common_score(all_sentences_words, title_words):
    set_title_words = set(title_words)
    ret = []
    for words in all_sentences_words:
        set_words = set(words)& set_title_words
        if len(set_words)>=3:
            ret.append(1.5)
        else:
            ret.append(1)
    return np.array(ret)

title_words = _seg_sent(title)# title_words 标题的词库
title_vec = get_sentnce_vector([title_words])# 得到标题中一个句子，一个向量
# title_sim_score为标题与每个内容句子的相似度（sentence_vec向量矩阵)title_sim_score =[[0.68385445],[0.75670076]...]
title_sim_score = get_title_similarity(sentence_vec, title_vec)
# 归一化 title_sim_score=[1.81772674 1.90483354 1.50085534...]
scaler = MinMaxScaler((1,2))
scaler.fit(title_sim_score)
title_sim_score = scaler.transform(title_sim_score)[:,0]
## 如果标题与内容句子有3个共有的token，赋值1.5，否则1 all_sentences_words（句子矩阵） title_common_score=[1  1.5  1  1  1.5  1... ]
title_common_score = get_title_common_score(all_sentences_words, title_words)


# ********** 计算各特征:3.首位置和末位更重要 ********** #
# 首位置和末位更重要
def get_position_score(sen_length):
    position_score = np.ones(sen_length)
    position_score[:3] = 2
    position_score[-3:] = 1.5
    return position_score
# 前3个赋值2，后3个赋值1.5，中间赋值1 position_score=[2.  2.  2.  1...]
position_score = get_position_score(len(sentences))


# ********** 计算各特征:4.是否包含结论性的词语 ********** #
# 含结论性的词语
def get_clue_score(sentences):
    clue_words = '总之 总而言之 综上 综上所述 一言以蔽之 概括起来说 括而言之 括而言之 要而论之 统而言之 归根到底 归根结底 简而言之'.split()
    result = []
    for sen in sentences:
        flag = 1
        for w in clue_words:
            if w in sen:
                flag = 1.4
                break
        result.append(flag)
    return np.array(result)
#包含结论词赋值1.4，否则1
clue_score = get_clue_score(sentences)


# ********** 计算各特征:5.综合各路特征score ********** #
title_common = False
# 句子特征的重要度：score=[0.03077326 0.03866788 0.01779828 0.00692144...]
score = pagerank_score * entities_score * (title_common_score if title_common else title_sim_score) * position_score * clue_score


# ********** MMR算法 ********** #
# 摘要数量
extract_num =  5

# 执行MMR算法
n = extract_num
summary_set = []
alpha = 0.8
# 取得句子特征重要度的最大值的下标
max_score_index = np.argmax(score)
summary_set.append(max_score_index)
while n > 0:
    # 计算所有句子与重要度最高的内容比相似度
    sim_mat = cosine_similarity(sentence_vec,sentence_vec[summary_set])  # [sent-size, sum-size]
    # 找一个每行最大的
    sim_mat = np.max(sim_mat,axis=1)  # [sent-size, 1]
    # 将句子特征值与和最重要句子的相似度结合起来
    feature_score = np.array([score, sim_mat]).T
    # 归一化
    scaler = MinMaxScaler()
    scaler.fit(feature_score)
    feature_score = scaler.transform(feature_score)
    [score, sim_mat] = feature_score[:,0], feature_score[:,1]
    # 计算mmr
    mmr_score =  alpha*score - (1-alpha)*sim_mat
    mmr_score[summary_set] = -100
    max_index  = np.argmax(mmr_score)
    summary_set.append(max_index)
    n -= 1


# ********** 获得摘要 ********** #
# 取摘要结果
summary = [sen for idx, sen in enumerate(sentences) if idx in summary_set]
print(summary)
