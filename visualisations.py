import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing as proc
from igraph import *


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

def visualise_gaph(g, layout = "circular"):
    # Set the layout
    visual_style = {}

    # Define colors used for outdegree visualization
    colours = ['#fecc5c', '#a31a1c']

    # Set bbox and margin
    visual_style["bbox"] = (600,500)
    visual_style["margin"] = 17
    visual_style["vertex_label_color"] = "red"

    # Set vertex colours
    visual_style["vertex_color"] = 'grey'

    # Set vertex size
    visual_style["vertex_size"] = 3
    visual_style["edge_width"] = 1

    # Set vertex lable size
    visual_style["vertex_label_size"] = 8

    # Don't curve the edges
    visual_style["edge_curved"] = True
    my_layout = g.layout(layout)
    visual_style["layout"] = my_layout
    return visual_style

def visualise_binomials(df, data_set, set_column, how_many, use_layout):
    size = df.shape[0]

    if set_column != "None":
        unique = df[set_column].value_counts()[: how_many].index.tolist()

        if unique[0] == "None":
            unique = df[set_column].value_counts()[1: how_many+1].index.tolist()
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


