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


def n_largest(arr, n):
    max_number = heapq.nlargest(n, arr)
    max_index = []
    for t in max_number:
        index = arr.index(t)
        max_index.append(index)
        arr[index] = 0
    return dict(zip(max_index, max_number))
