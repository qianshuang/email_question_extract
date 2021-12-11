# -*- coding: utf-8 -*-

import string
from nltk import sent_tokenize

import spacy

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型

s = "Dear Sir or Madam, I just learned about your Bank on Facebook.Do you have a physical branch or are you branchless and just an online bank?"
sent_tokenize_list_ori = sent_tokenize(s)
print(sent_tokenize_list_ori)
#
# test_doc = nlp(s)
# for sent in test_doc.sents:
#     print(str(sent))
