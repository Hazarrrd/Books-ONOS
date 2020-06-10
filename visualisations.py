import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

import data_preprocessing as proc
from igraph import *


def discretize_data(data, strategy="uniform"):
    data_disc = [[i] for i in data]
    # kmeans/quantile/uniform
    enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=strategy)
    enc.fit(data_disc)
    grid_encoded = enc.transform(data_disc)
    print(enc.bin_edges_)
    return [int(j) + 1 for sub in grid_encoded for j in sub]


def make_feature_matrix(size, feature):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if feature[i] == feature[j] and feature[i] != -1:
                matrix[i][j] = 1
                matrix[j][i] = 1
    for i in range(size):
        matrix[i][i] = 0
    return matrix


def visualise_gaph(g, layout="circular"):
    # Set the layout
    visual_style = {}

    # Define colors used for outdegree visualization
    colours = ['#fecc5c', '#a31a1c']

    # Set bbox and margin
    visual_style["bbox"] = (700, 700)
    visual_style["margin"] = 100
    visual_style["vertex_label_color"] = "Red"
    # visual_style["vertex_label_color"] = "Dark Red"

    # Set vertex colours
    visual_style["vertex_color"] = 'grey'

    # Set vertex size
    visual_style["vertex_size"] = 3
    visual_style["edge_width"] = 1

    # Set vertex lable size
    visual_style["vertex_label_size"] = 8

    # Don't curve the edges
    visual_style["edge_curved"] = True
    visual_style["vertex_label_size"] = 8
    my_layout = g.layout(layout)
    visual_style["layout"] = my_layout
    return visual_style


def visualise_binomials(df, data_set, set_column, how_many, use_layout):
    size = df.shape[0]

    if set_column != "None":
        unique = df[set_column].value_counts()[: how_many].index.tolist()

        if unique[0] == "None":
            unique = df[set_column].value_counts()[1: how_many + 1].index.tolist()
    else:
        unique = list(set(data_set))

    dic_val_unique = {aut: num + size for num, aut in enumerate(unique)}

    edges = []
    for i in range(size):
        if i < len(data_set) and data_set[i] in dic_val_unique.keys():
            edges.append((i, dic_val_unique[data_set[i]]))

    g = Graph.Bipartite([0] * size + [1] * len(dic_val_unique), edges)

    labels = [""] * len(g.vs)
    labels[len(g.vs) - len(unique):] = unique
    g.vs["label"] = labels
    # delete books without connections with its author
    to_delete_ids = [v.index for v in g.vs if v.degree() <= 0]
    g.delete_vertices(to_delete_ids)

    if use_layout == "Bipartite":
        visual_style = visualise_gaph(g)
        visual_style["layout"] = g.layout_bipartite()
    else:
        visual_style = visualise_gaph(g, use_layout)

    return g, visual_style


def visualise_tags_binomial(tag_limit, use_layout, how_many_books):
    df, shelves = proc.get_all_data(tag_limit)
    size = df.shape[0]

    all_tags = []
    all_keys = []
    for key, value in shelves.items():
        all_tags += value
        all_keys.append(key)

    unique = list(set(all_tags))
    dic_val_unique = {aut: num + size for num, aut in enumerate(unique)}
    dic_key_unique = {aut: num for num, aut in enumerate(all_keys)}

    edges = []
    counter = 0
    limit = how_many_books
    for key, item in shelves.items():
        if counter > limit:
            break
        for tag in item:
            edges.append((dic_key_unique[key], dic_val_unique[tag]))
        counter += 1
    g = Graph.Bipartite([0] * size + [1] * len(dic_val_unique), edges)
    labels = [""] * len(g.vs)
    labels[len(g.vs) - len(unique):] = unique
    g.vs["label"] = labels

    # delete books without connections with its author
    to_delete_ids = [v.index for v in g.vs if v.degree() <= 0]
    g.delete_vertices(to_delete_ids)

    if use_layout == "Bipartite":
        visual_style = visualise_gaph(g)
        visual_style["layout"] = g.layout_bipartite()
    else:
        visual_style = visualise_gaph(g, use_layout)

    return g, visual_style


def visualise_normal_graph(data_set, degree, use_layout):
    size = len(data_set)
    matrix = make_feature_matrix(size, data_set)

    g = Graph.Adjacency((matrix > 0).tolist(), mode=ADJ_MAX)
    # show only books with degree bigger than our parametr 'degree'
    to_delete_ids = [v.index for v in g.vs if v.degree() <= degree]
    g.delete_vertices(to_delete_ids)

    visual_style = visualise_gaph(g, use_layout)
    return g, visual_style


def visualuse_similarity(use_layout, how_many, tag_limit, ids, scal, titles, authors,
                         matrix_authors, matrix_years, matrix_pages,
                         matrix_rates, matrix_popularity, title_list=None):
    df, shelves = proc.get_all_data(tag_limit)
    size = df.shape[0]

    correct = False
    while not correct:
        inx = np.random.choice(size, how_many, replace=False)
        correct = True
        for i in inx:
            if str(ids[i]) not in shelves.keys():
                correct = False

    if title_list:
        inx = [titles.index(title) for title in title_list]
        how_many = len(inx)

    matrix_shelves = np.array([[len(set(shelves[str(ids[i])]) & set(shelves[str(ids[j])])) for i in inx] for j in inx])

    matrix = matrix_authors * scal["auth"] + matrix_years * scal["years"] + matrix_pages * scal["pages"] \
             + matrix_rates * scal["rates"] + matrix_popularity * scal["popularity"]

    matrix = np.array([[matrix[j][i] for i in inx] for j in inx])
    matrix += matrix_shelves * scal["shelves"]
    sum_weight = 0
    for i in range(how_many):
        for j in range(how_many):
            if i >= j:
                matrix[i][j] = 0
            else:
                sum_weight += matrix[i][j]

    g = Graph.Adjacency((matrix > 0).tolist(), mode=ADJ_MAX)

    g.es['weight'] = matrix[matrix.nonzero()]
    g.vs['label'] = [titles[i] + "\n~" + authors[i] for i in inx]
    # to_delete_ids = [v.index for v in g.vs if v.degree() <= degree]
    # g.delete_vertices(to_delete_ids)ut

    visual_style = visualise_gaph(g, use_layout)
    # visual_style["edge_width"] = [150*is_formal/sum_weight for is_formal in g.es["weight"]]
    visual_style["edge_width"] = [(10 * how_many * is_formal / sum_weight) for is_formal in g.es["weight"]]

    colors = {1: "Pink", 2: "Deep Pink", 3: "Purple", 4: "Navy Blue", 5: "Black"}
    disc_data = discretize_data(g.es['weight'])
    g.es["color"] = [colors[edge_disc] for edge_disc in disc_data]
    return g, visual_style
