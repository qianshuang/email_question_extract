# -*- coding: utf-8 -*-

from common import *

# import spacy
# import neuralcoref
# from nltk import sent_tokenize

# nlp = spacy.load('en_core_web_sm')  # 加载预训练模型
# neuralcoref.add_to_pipe(nlp)

# from bert_common import *

# from graph_common import *

# from textrank4zh import TextRank4Keyword, TextRank4Sentence

# from bertopic import BERTopic
# from nltk import sent_tokenize
# from textteaser import TextTeaser

# from pyhanlp import *
# from bosonnlp import BosonNLP

# from top2vec import Top2Vec

# s = """Hello Adam, thanks for getting back to me. I need a savings account for my business sales and to receive payments from my online customers. What are the requirements? Thanks in advance, Jes"""
# docs = sent_tokenize(s)
# print(docs)

# topic_model = BERTopic(min_topic_size=1)
# topics, probs = topic_model.fit_transform(docs)
# print(topics)
# print(probs)

# model = Top2Vec(docs)
# topic_sizes, topic_nums = model.get_topic_sizes()
# print(topic_sizes)
# print(topic_nums)
#
# documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=48, num_docs=5)
# for doc, score, doc_id in zip(documents, document_scores, document_ids):
#     print(f"Document: {doc_id}, Score: {score}")
#     print("-----------")
#     print(doc)
#     print("-----------")
#     print()

# text = s
# tr4s = TextRank4Sentence()
# tr4s.analyze(text=text, lower=True, source='all_filters')
# for item in tr4s.get_key_sentences(num=3):
#     print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重

delete_sents = ["To whom it may concern"]
Incoming_email_content = "To whom it may concern, Do you have Third Currency Accounts?  I'd like to open a Canadian Account. May I also know what other currencies can I open with you? Can I earn an interest with these accounts? What requirements do you need? Please respond at your earliest convenience. Thank you, Jes"
Incoming_email_content = Incoming_email_content.lower()
for ds in delete_sents:
    Incoming_email_content = Incoming_email_content.replace(ds.lower(), "").strip()
print(Incoming_email_content)
