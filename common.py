# -*- coding: utf-8 -*-

import heapq
import numpy as np


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    return [line.strip() for line in open(filename).readlines()]


def write_file(filename, content):
    open_file(filename, mode="w").write(content)


def write_lines(filename, list_res):
    test_w = open_file(filename, mode="w")
    for j in list_res:
        test_w.write(j + "\n")


def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    # return num / denom


# 余弦距离矩阵：不允许负数出现
def cos_distance_metric(vecs):
    cos_dis_metric = np.zeros([len(vecs), len(vecs)])
    for i in range(len(vecs)):
        cos_dis_metric[i][i] = 1
        for j in range(i + 1, len(vecs)):
            cos_dis = get_cos_similar(vecs[i], vecs[j])
            cos_dis_metric[i][j] = cos_dis
            cos_dis_metric[j][i] = cos_dis
    return cos_dis_metric


def n_largest(arr, n):
    max_number = heapq.nlargest(n, arr)
    max_index = []
    for t in max_number:
        index = arr.index(t)
        max_index.append(index)
        arr[index] = 0
    return dict(zip(max_index, max_number))


def fill_pre(sent_tokenize_list_ori, ranked_sentences, sent_indexes, bert_sent_vecs, need_pre_sents_vecs):
    ranked_sentences_new = []
    for i in range(len(bert_sent_vecs)):
        app_flag = True
        bsv = bert_sent_vecs[i]
        for npsv in need_pre_sents_vecs:
            if get_cos_similar(bsv, npsv) >= 0.9:
                ranked_sentences_new.append((ranked_sentences[i][0],
                                             sent_tokenize_list_ori[sent_indexes[i] - 1] + " " + ranked_sentences[i][
                                                 1]))
                app_flag = False
                break
        if app_flag:
            ranked_sentences_new.append(ranked_sentences[i])
    return ranked_sentences_new


def fill_non_ques(non_ques_vecs, subject_bert_vec, Incoming_email_subject, declarative_words):
    ranked_sentences = []
    if Incoming_email_subject == "":
        return ranked_sentences

    for idx_vec in non_ques_vecs:
        flag = False
        for dw in declarative_words:
            if dw in idx_vec[0]:
                flag = True
                break
        if flag:
            sim_scr = get_cos_similar(idx_vec[1], subject_bert_vec)
            if sim_scr >= 0.8:
                ranked_sentences.append((sim_scr, idx_vec[0]))
    return ranked_sentences
