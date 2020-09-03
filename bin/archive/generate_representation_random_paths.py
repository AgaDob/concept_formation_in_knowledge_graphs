#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script:
# 1) Represents each entity using a vector of N, where N is the number of random paths.
# Either counting or binary indication for when given entitity participates as 1) a subject or 2) an object with the given relation.
# 2) Saves to embeddings matrix
# Note: here we treat the graph an **undirectional** and make **no distinction** between ingoing and outgoing edges


import os
import sys
import argparse
import logging
import os.path

import multiprocessing
import numpy as np
import pandas as pd
from collections import Counter


from metakbc.training.data import Data


def get_adjacencies(data):
  # get all triples associated with a given entity
  print('Getting adjacency data...')
  all_triples = data.X
  train_adj_dict = {key:[] for key in list(data.idx_to_entity.keys())}
  for i in range(data.X.shape[0]):
    triple = list(all_triples[i, :])
    s, p, o = triple

    # forward relation
    adj_list = train_adj_dict[s]
    adj_list.append([s, p, o])
    train_adj_dict[s] = adj_list

    # inverse relation
    adj_list = train_adj_dict[o]
    adj_list.append([o, p, s])
    train_adj_dict[o] = adj_list

  train_adj_list = list(train_adj_dict.values())
  return train_adj_dict, train_adj_list


def check_for_loops(path):
  # Check for Loops: Have the last 3 consecutive segments already appeared in this path?
  seg = path[-3:]

  for i in range(len(path) - 3):
    if path[i] == seg[0] and path[i+1] == seg[1] and path[i+2] == seg[2]:
      print('Loop detected!')
      return True

  return False


def get_paths(data, train_adj_list, n_paths=200, min_path_len=3, max_path_len=100):
  print('Getting paths...')
  all_paths = []

  if min_path_len <= max_path_len:
      max_path_len= min_path_len + 1 # to avoide Value Error
  print('min_path_len: ' + str(min_path_len))
  print('max_path_len: ' + str(max_path_len))

  for j in range(n_paths):
    path_len = np.random.choice(range(min_path_len, max_path_len))
    path = []
    prev_rel = None

    # initialise start node
    curr_node =  np.random.choice(range(data.nb_entities))
    outgoing_edges = train_adj_list[curr_node]

    while len(outgoing_edges) == 0:
      curr_node =  np.random.choice(range(data.nb_entities))
      outgoing_edges = train_adj_list[curr_node]

    while len(path) < path_len:
     
      if len(path) > 6:
        if check_for_loops(path):
          break

      # pick one at random
      out_edge_idx = np.random.choice(range(len(outgoing_edges)))
      out_edge = outgoing_edges[out_edge_idx]

      path.append(out_edge)

      prev_rel = out_edge[1]
      curr_node = out_edge[2]


    path  = [tuple(x) for x in path]
    set_path = set(tuple(x) for x in path)
    unique_path = [x for x in path if x in set_path]
    all_paths.append(unique_path)

  return all_paths


def get_X(all_paths, data, n_paths):
  print('Generating matrix of embeddings...')
  # generate vector representation for each entity
  X = np.zeros(shape=(data.nb_entities, n_paths))
  for path_n, path in enumerate(all_paths):

    print('path length: ' + str(len(path))) # debugging
    ent_indx = np.array(path)[:,0]

    cnt = Counter(ent_indx)
    entities = list(cnt.keys())
    ent_count = list(cnt.values())

    X[entities, path_n] += ent_count

  return X





def main(argv):
    parser = argparse.ArgumentParser('Meta-KBC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--data_name', action='store', type=str, default=None)
    parser.add_argument('--embeddings_path', action='store', type=str, default=None)
    parser.add_argument('--n_paths', action='store', type=int, default=200)
    parser.add_argument('--min_path_len', action='store', type=int, default=3)
    parser.add_argument('--max_path_len', action='store', type=int, default=100)


    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    embeddings_path = args.embeddings_path
    data_name = args.data_name
    n_paths = args.n_paths
    min_path_len = args.min_path_len
    max_path_len = args.max_path_len

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    test_i_path = args.test_i
    test_ii_path = args.test_ii
    input_type = args.input_type

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                    test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    train_adj_dict, train_adj_list = get_adjacencies(data)
    all_paths = get_paths(data, train_adj_list, n_paths=n_paths, min_path_len=min_path_len, max_path_len=max_path_len)
    X = get_X(all_paths, data, n_paths)
    np.savetxt(embeddings_path, X, fmt='%d')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
