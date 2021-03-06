# -*- coding: utf-8 -*-

from config import *
import pandas as pd
from nltk import sent_tokenize
import string

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
    Incoming_email_subject = "" if pd.isnull(row["Incoming email subject"]) else row[
        "Incoming email subject"].strip().lower()
    for c in string.punctuation:
        Incoming_email_subject = Incoming_email_subject.replace(c, "")
    subject_bert_vec = get_bert_sent_vecs([Incoming_email_subject])[0]

    Incoming_email_content = "" if pd.isnull(row["Incoming email content"]) else row["Incoming email content"]
    Incoming_email_content = Incoming_email_content.replace("this", "it")
    Incoming_email_content = Incoming_email_content.replace("that", "it")
    Incoming_email_content = Incoming_email_content.replace("them", "it")
    Incoming_email_content = Incoming_email_content.replace("these", "it")
    Incoming_email_content = Incoming_email_content.replace("those", "it")
    for ds in delete_sents:
        Incoming_email_content = Incoming_email_content.replace(ds, "")
        Incoming_email_content = Incoming_email_content.replace(ds.lower(), "")
    Incoming_email_content = Incoming_email_content.strip().strip(string.punctuation).strip()

    doc = nlp(Incoming_email_content)
    # 1. 指代消歧
    Incoming_email_content = doc._.coref_resolved
    # 2. 去除人名
    for ent in doc.ents:
        print(ent.text, ent.label_)  # Jes PERSON
        if ent.label_ == "PERSON":
            Incoming_email_content = Incoming_email_content.replace(ent.text, "")
        # if ent.label_ == "ORG":
        #     Incoming_email_content = Incoming_email_content.replace(ent.text, "")

    Incoming_email_content = Incoming_email_content.strip().strip(string.punctuation).strip().lower()
    sent_tokenize_list_ori = sent_tokenize(Incoming_email_content)
    # 只提取问句
    sent_indexes = [i for i in range(len(sent_tokenize_list_ori)) if sent_tokenize_list_ori[i].strip()[-1] == "?"]
    sent_tokenize_list = [s.strip() for s in sent_tokenize_list_ori if s.strip()[-1] == "?"]
    if len(sent_tokenize_list) == 0:
        all_qs.append("")
        continue
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
        for ssv in stop_sents_vecs:
            if get_cos_similar(bert_sent_vecs[i], ssv) >= 0.9:
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
    none_ques_ranked_sentences = fill_non_ques(non_ques_vecs, subject_bert_vec, Incoming_email_subject,
                                               declarative_words)
    ranked_sentences = sorted(ranked_sentences + none_ques_ranked_sentences, reverse=True)
    all_qs.append(str(ranked_sentences))

df_final["extract_questions"] = all_qs
print(df_final)
df_final.to_csv('data/email_question_result_mark.txt', sep='\t', index=False)
