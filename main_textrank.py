# -*- coding: utf-8 -*-

from common import *
from bert_common import *
from graph_common import *
import pandas as pd
import spacy
import neuralcoref
from nltk import sent_tokenize
import networkx as nx
import string

from sklearn.metrics.pairwise import cosine_similarity

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

all_cols = ["Category", "Topic", "Incoming email subject", "Incoming email content"]
df_final = pd.DataFrame()
df = pd.read_excel("data/banking_emails.xlsx", engine="openpyxl", sheet_name=None)
for k, v in df.items():
    if k == "summary":
        continue
    v_df = v.loc[:, all_cols]
    df_final = pd.concat([df_final, v_df], ignore_index=True)

df_final = df_final.dropna(subset=["Incoming email content"])  # 81行
df_final['Incoming email content'] = df_final['Incoming email content'].apply(
    lambda x: str(x).replace('\t', ' ').replace('\n', ' '))
print(df_final.head())
print(df_final.shape)

all_qs = []
for index, row in df_final.iterrows():
    Incoming_email_content = row["Incoming email content"]
    Incoming_email_subject = row["Incoming email subject"]

    sent_list = []
    if not pd.isnull(Incoming_email_subject) and Incoming_email_subject.strip() != "":
        sent_list.append(Incoming_email_subject)
    if not pd.isnull(Incoming_email_content) and Incoming_email_content.strip() != "":
        Incoming_email_content = Incoming_email_content.lower()
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

        sent_tokenize_list = sent_tokenize(Incoming_email_content)
        # 3. 去除标点
        sent_tokenize_list_ = []
        for s in sent_tokenize_list:
            for c in string.punctuation:
                s = s.replace(c, "")
            if s.strip() != "":
                sent_tokenize_list_.append(s.strip())
        sent_tokenize_list = sent_tokenize_list_
        sent_list = sent_list + sent_tokenize_list

    sent_tokenize_list_ori = sent_list
    bert_sent_vecs = get_bert_sent_vecs(sent_list)
    # 4. 删除停用句
    index_to_delete = []
    for i in range(len(bert_sent_vecs)):
        for ssv in stop_sents_vecs:
            if get_cos_similar(bert_sent_vecs[i], ssv) >= 0.9:
                index_to_delete.append(i)
                break
    bert_sent_vecs = [bert_sent_vecs[i] for i in range(len(bert_sent_vecs)) if i not in index_to_delete]
    sent_list = [sent_list[i] for i in range(len(sent_list)) if i not in index_to_delete]
    sent_indexes = [i for i in range(len(sent_list)) if i not in index_to_delete]

    # cos_simi_m = cosine_similarity(bert_sent_vecs)
    cos_simi_m = cos_distance_metric(bert_sent_vecs)
    # nx_graph = nx.from_numpy_array(cos_simi_m)
    nx_graph = build_graph(bert_sent_vecs, cos_simi_m)

    scores = nx.pagerank(nx_graph)
    if len(sent_list) == 1:
        scores[0] = 1

    ranked_sentences = [(scores[i], s) for i, s in enumerate(sent_list)]
    ranked_sentences = fill_pre(sent_tokenize_list_ori, ranked_sentences, sent_indexes, bert_sent_vecs,
                                need_pre_sents_vecs)
    ranked_sentences = [item for item in ranked_sentences if
                        len(ranked_sentences) > 0 and item[0] >= 1 / len(ranked_sentences)]
    ranked_sentences = sorted(ranked_sentences, reverse=True)
    all_qs.append(str(ranked_sentences))

df_final["extract_questions"] = all_qs
print(df_final)
df_final.to_csv('data/email_question_result_textrank.txt', sep='\t', index=False)
