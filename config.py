# -*- coding: utf-8 -*-

from common import *
from bert_common import *
import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型
neuralcoref.add_to_pipe(nlp)

# 加载停用句
stop_sents = read_file("resources/stop_sents.txt")
stop_sents_vecs = get_bert_sent_vecs(stop_sents)
print("compute stop sentences bert vectors finished...")

# 加载无用客套语
delete_sents = read_file("resources/delete_sents.txt")

# 加载需前向查找语
need_pre_sents = read_file("resources/need_pre_sents.txt")
need_pre_sents_vecs = get_bert_sent_vecs(need_pre_sents)
print("compute need_pre sentences bert vectors finished...")

# 加载陈述句诉求词
declarative_words = read_file("resources/declarative_words.txt")
