# -*- coding: utf-8 -*-

from common import *
from bert_common import *
import spacy
import neuralcoref
from nltk import sent_tokenize
import string

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型
neuralcoref.add_to_pipe(nlp)

# 加载停用句
stop_sents = read_file("resources/stop_sents.txt")
stop_sents_vecs = get_bert_sent_vecs(stop_sents)

# 加载无用客套语
delete_sents = read_file("resources/delete_sents.txt")

# 加载需前向查找语
need_pre_sents = read_file("resources/need_pre_sents.txt")
need_pre_sents_vecs = get_bert_sent_vecs(need_pre_sents)

all_qs = []
Incoming_email_subject = ""
subject_bert_vec = get_bert_sent_vecs([Incoming_email_subject])[0]

Incoming_email_content = "Hi Adam, Thank you for your reply. Couldn't it be delivered to our house? I am now 50 years old and could not come to your office. Thank you, Jes".strip().lower()
for ds in delete_sents:
    Incoming_email_content = Incoming_email_content.replace(ds.lower(), "").strip()

doc = nlp(Incoming_email_content)
# 1. 指代消歧
Incoming_email_content = doc._.coref_resolved
# 2. 去除人名
for ent in doc.ents:
    print(ent.text, ent.label_)  # Jes PERSON
    if ent.label_ == "PERSON":
        Incoming_email_content = Incoming_email_content.replace(ent.text, "")

sent_tokenize_list_ori = sent_tokenize(Incoming_email_content)
# 只提取问句
sent_indexes = [i for i in range(len(sent_tokenize_list_ori)) if sent_tokenize_list_ori[i].strip()[-1] == "?"]
sent_tokenize_list = [s.strip() for s in sent_tokenize_list_ori if s.strip()[-1] == "?"]

# 3. 去除标点
sent_tokenize_list_ = []
for s in sent_tokenize_list:
    for c in string.punctuation:
        s = s.replace(c, "")
    if s.strip() != "":
        sent_tokenize_list_.append(s.strip())
sent_tokenize_list = sent_tokenize_list_

bert_sent_vecs = get_bert_sent_vecs(sent_tokenize_list)
# 4. 删除停用句
index_to_delete = []
for i in range(len(bert_sent_vecs)):
    for j in range(len(stop_sents_vecs)):
        ssv = stop_sents_vecs[j]
        if get_cos_similar(bert_sent_vecs[i], ssv) >= 0.9:
            print(stop_sents[j])
            index_to_delete.append(i)
            break
bert_sent_vecs = [bert_sent_vecs[i] for i in range(len(bert_sent_vecs)) if i not in index_to_delete]
sent_list = [sent_tokenize_list[i] for i in range(len(sent_tokenize_list)) if i not in index_to_delete]
sent_indexes = [sent_indexes[i] for i in range(len(sent_indexes)) if i not in index_to_delete]

scores = {}
for i in range(0, len(bert_sent_vecs)):
    if Incoming_email_subject == "":
        scores[i] = 0
    else:
        scores[i] = get_cos_similar(bert_sent_vecs[i], subject_bert_vec)
ranked_sentences = [(scores[i], s) for i, s in enumerate(sent_list)]
ranked_sentences = fill_pre(sent_tokenize_list_ori, ranked_sentences, sent_indexes, bert_sent_vecs,
                            need_pre_sents_vecs)
non_ques_vecs = [(sent_tokenize_list_ori[i], get_bert_sent_vecs([sent_tokenize_list_ori[i]])[0]) for i in
                 range(len(sent_tokenize_list_ori)) if i not in sent_indexes]
none_ques_ranked_sentences = fill_non_ques(non_ques_vecs, subject_bert_vec, Incoming_email_subject)
ranked_sentences = sorted(ranked_sentences + none_ques_ranked_sentences, reverse=True)
all_qs.append(str(ranked_sentences))

print(all_qs)
