#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script:
# 1) Contains all of the Weisfeiler-Lehman Kernel function defintions (later to be moved into the models folder).
# The WL Kernel below is an implementation by Lorenzo Palloni and Emilio Cecchini (https://github.com/deeplego/wl-graph-kernels)
# based on the 'A fast approximation of the Weisfeiler-Lehman Graph Kernels' paper.
# 2) The resultant kernel is saved to the specified embeddings_path


import rdflib
import nptyping
from rdflib import Graph
from typing import (
    List,
    Dict,
    Tuple,
    Iterable,
    Union,
    Set,
)
from itertools import chain
from nptyping import NDArray as Array

import os
import sys
import argparse
import logging
import os.path

import multiprocessing
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm


class Node:
    'A node of a Weisfeiler-Lehman RDF graph'

    def __init__(self):
        self.neighbors = set()

    def add_neighbor(self, edge):
        self.neighbors.add(edge)

    def __hash__(self):
        return hash(id(self))


class Edge:
    'An edge of a Weisfeiler-Lehman RDF graph'

    def __init__(self):
        self.neighbor = None

    def __hash__(self):
        return hash(id(self))




class WLRDFGraph:
    'Weisfeiler-Lehman RDF graph'

    def __init__(self, triples: Iterable[Tuple[str, str, str]],
                 instances: Iterable[str], max_depth: int):
        'Build a Weisfeiler-Lehman RDF graph from a list of RDF triples'
        triples = list(triples)
        self.max_depth = max_depth
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()
        self.labels: List[Dict[Tuple[Union[Node, Edge], int], str]] = [dict()]
        self.instance_nodes: Dict[str, Dict[Node, int]] = {
            instance: dict() for instance in instances
        }
        self.instance_edges: Dict[str, Dict[Edge, int]] = {
            instance: dict() for instance in instances
        }

        v_map: Dict[str, Node] = dict()
        e_map: Dict[Tuple[str, str, str], Edge] = dict()

        # 1. Initialization
        print('Initialising graph ...')
        for instance in instances:
            root = Node()
            self.nodes.add(root)
            self.labels[0][(root, max_depth)] = 'root'
            v_map[instance] = root

        # 2. Subgraph Extraction
        print('Beginning subgraph extraction')
        ct = 0
        total_inst = len(instances)


        for nb, instance in enumerate(instances):

            ct +=1
            search_front = {instance}
            for j in reversed(range(0, max_depth)):
                new_search_front = set()
                for r in search_front:
                    r_triples = ((s, p, o) for s, p, o in triples if s == r)
                    for sub, pred, obj in r_triples:
                        new_search_front.add(obj)

                        if obj not in v_map:
                            v = Node()
                            self.nodes.add(v)
                            v_map[obj] = v
                        self.labels[0][(v_map[obj], j)] = obj
                        if v_map[obj] not in self.instance_nodes[instance]:
                            self.instance_nodes[instance][v_map[obj]] = j

                        t = (sub, pred, obj)
                        if t not in e_map:
                            e = Edge()
                            self.edges.add(e)
                            e_map[t] = e
                        self.labels[0][e_map[t], j] = pred
                        if e_map[t] not in self.instance_edges[instance]:
                            self.instance_edges[instance][e_map[t]] = j

                        v_map[obj].add_neighbor(e_map[t])
                        e_map[t].neighbor = v_map[sub]

                search_front = new_search_front




    def relabel(self, iterations: int = 1):
        'Relabeling algorithm'

        for i in range(len(self.labels), len(self.labels) + iterations):

            multisets: Dict[Tuple[Union[Node, Edge], int], List[str]] = dict()

            # 1. Multiset-label determination
            for v in self.nodes:
                for j in range(self.max_depth + 1):
                    if (v, j) in self.labels[0]:
                        multisets[(v, j)] = [
                            self.labels[i - 1][(u, j)] for u in v.neighbors
                            if (u, j) in self.labels[i - 1]
                        ]
            for e in self.edges:
                for j in range(self.max_depth):
                    if (e, j) in self.labels[0]:
                        multisets[(e, j)] = [
                            self.labels[i - 1][(e.neighbor, j + 1)]
                        ]

            # 2. Sorting each multiset
            expanded_labels = {
                (k, j): self.labels[i - 1][(k, j)] + ''.join(sorted(multiset))
                for (k, j), multiset in multisets.items()
            }

            # 3. Label compression
            f = {
                s: str(i)
                for i, s in enumerate(set(expanded_labels.values()))
            }

            # 4. Relabeling
            self.labels.append({
                (k, j): f[expanded_labels[(k, j)]]
                for (k, j) in expanded_labels
            })


def count_commons(a: Iterable, b: Iterable) -> int:
    'Return the number of common elements in the two iterables'
    uniques = set(a).intersection(set(b))
    counter_a = Counter(a)
    counter_b = Counter(b)
    commons = 0
    for u in uniques:
        commons += counter_a[u] * counter_b[u]
    return commons


def wlrdf_kernel(graph: WLRDFGraph, instance_1: str, instance_2: str,
                 iterations: int = 0) -> float:
    'Compute the Weisfeiler-Lehman kernel for two instances'

    if iterations > len(graph.labels) - 1:
        graph.relabel(iterations - len(graph.labels) + 1)

    kernel = 0.0
    for it in range(iterations + 1):
        node_labels_1 = [
            graph.labels[it][(v, d)]
            for v, d in graph.instance_nodes[instance_1].items()
        ]
        node_labels_2 = [
            graph.labels[it][(v, d)]
            for v, d in graph.instance_nodes[instance_2].items()
        ]
        edge_labels_1 = [
            graph.labels[it][(e, d)]
            for e, d in graph.instance_edges[instance_1].items()
        ]
        edge_labels_2 = [
            graph.labels[it][(e, d)]
            for e, d in graph.instance_edges[instance_2].items()
        ]
        cc_nodes = count_commons(node_labels_1, node_labels_2)
        cc_edges = count_commons(edge_labels_1, edge_labels_2)
        w = (it + 1) / (iterations + 1)
        kernel += w * (cc_nodes + cc_edges)
    return kernel



def wlrdf_kernel_matrix(graph: WLRDFGraph, instances: List[str],
                        iterations: int = 0) -> Array[float]:
    'Compute the matrix of the kernel values between each couple of instances'
    n = len(instances)
    kernel_matrix = np.zeros((n, n))



    # stats
    print('Kernel size: {} by {}'.format(n,n))
    diagonal = n
    total = n*n
    upper_triangle = (total - diagonal)/2
    to_compute = int(upper_triangle + diagonal)
    print('Total number of computations to be performed: {}'.format(to_compute))
    count = 0

    for i in range(n):
        for j in range(i, n):
            kernel_matrix[i][j] = wlrdf_kernel(
                graph, instances[i], instances[j], iterations
            )
            count+=1
            if count % 100000 ==0:
                print('{}% complete'.format(np.round(count/to_compute*100,4)))
                print('Sanity check - random kernel value: {}'.format(str(kernel_matrix[i][j])))



    print('Computing lower left triangle...')
    for i in range(n):
        for j in range(0, i):
            kernel_matrix[i][j] = kernel_matrix[j][i]
    return kernel_matrix


def kernel_normalization(kernel_matrix: Array[float]) -> Array[float]:
    n = kernel_matrix.shape[0]
    res = np.zeros((n, n))
    assert kernel_matrix.shape[1] == n
    for i in range(n):
        for j in range(n):
            res[i][j] = kernel_matrix[i][j] / np.sqrt(
                kernel_matrix[i][i] * kernel_matrix[j][j]
            )
    return res



def main(argv):
    parser = argparse.ArgumentParser('Meta-KBC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # here train is a train.ttl file
    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--data_name', action='store', type=str, default=None)
    parser.add_argument('--embeddings_path', action='store', type=str, default=None)

    # kernel arguments
    parser.add_argument('--max_depth_subgraph', action='store', type=int, default=3)
    parser.add_argument('--iterations', action='store', type=int, default=6)
    parser.add_argument('--normalisation', action='store', type=str, default=None)

    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    embeddings_path = args.embeddings_path
    data_name = args.data_name
    max_depth_subgraph = args.max_depth_subgraph
    iterations = args.iterations
    train_path = args.train

    normalisation = args.normalisation


    ### GET KERNEL ###


    rdf_graph = Graph().parse(train_path, format='turtle')

    instances_class_map = {str(s): str(o) for s, p, o in rdf_graph }
    instances = list(instances_class_map.keys())
    triples = list((str(s), str(p), str(o)) for s, p, o in rdf_graph)

    print('Creating WLRDFGraph object...')
    wlrdf_graph = WLRDFGraph(triples, instances, max_depth=max_depth_subgraph)

    print('Computing kernel matrix...')
    K = wlrdf_kernel_matrix(wlrdf_graph, instances, iterations=iterations)

    if normalisation is not None:
        print('Beginning Matrix Normalisation...')
        K = kernel_normalization(K)



    print('Exporting embeddings...')
    np.savetxt(embeddings_path, K, fmt='%d')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
