# -*- coding: utf-8 -*-

import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型
neuralcoref.add_to_pipe(nlp)

s = """Dear Sir or Madam,  I used to register +1 805 2354 as my contact number. But now I have a new number +1 525 6666 and I want to use this as my contact number regarding my accounts. Could you please update it for me? Thanks, Jes"""
doc = nlp(s)
print(doc._.coref_resolved)
