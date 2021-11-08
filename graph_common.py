# -*- coding: utf-8 -*-

import networkx as nx

simi_threshold = 0


def build_graph(questions_vecs, cos_simi_m):
    G = nx.Graph()
    for i in range(len(questions_vecs)):
        for j in range(i + 1, len(questions_vecs)):
            if cos_simi_m[i][j] >= simi_threshold:
                G.add_edge(i, j, weight=cos_simi_m[i][j])
    return G
