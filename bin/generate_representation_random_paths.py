#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script:
# 1) Represents each entity using a vector of 2P, where P is the number of relations in the dataset.
# 2) For each entity e_i, it generates N random paths starting at e_i
# 3) Next, the the relations which occur during the random path (distinguishing between 'forward' and 'inverse') are counted
# 4) And the resulting embeddings are saved to an embeddings matrix

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


from metakbc.training.data import Data



def check_for_loops(path):
  # Check for Loops: Have the last 3 consecutive segments already appeared in this path?
  seg = path[-3:]

  for i in range(len(path) - 3):
    if path[i] == seg[0] and path[i+1] == seg[1] and path[i+2] == seg[2]:
      print('Loop detected!')
      return True

  return False



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
    adj_list.append([s, p, o, 1]) # 1 for forward
    train_adj_dict[s] = adj_list

    # inverse relation
    adj_list = train_adj_dict[o]
    adj_list.append([o, p, s, -1]) # -1 for inverse
    train_adj_dict[o] = adj_list

  train_adj_list = list(train_adj_dict.values())
  return train_adj_dict, train_adj_list



def get_paths(train_adj_list, train_adj_dict, n_paths=200, min_path_len=3, max_path_len=100):
  print('\nGetting paths...')

  entities = list(train_adj_dict.keys())
  all_paths = {key:[] for key in entities}


  # to avoide Value Error
  if min_path_len >= max_path_len:
      max_path_len= min_path_len + 1


  for start_node in tqdm(entities):

    for j in range(n_paths):
      path = []
      prev_rel = None

      # initialise start node
      curr_node = start_node

      for l in range(max_path_len):
        outgoing_edges = train_adj_list[curr_node].copy()


        # make sure we dont take inv of a prev edge
        if prev_rel is not None:
            prev_inverse = [curr_node, prev_rel, prev_node, - direction]
            outgoing_edges.remove(prev_inverse)

        if len(outgoing_edges) == 0:
          break


        if len(path) > 6:
          if check_for_loops(path):
            break

        # pick one at random
        out_edge_idx = np.random.randint(0, high=len(outgoing_edges))
        out_edge = outgoing_edges[out_edge_idx]

        path.append(out_edge)

        prev_node = out_edge[0]
        prev_rel = out_edge[1]
        curr_node = out_edge[2]
        direction = out_edge[3]


      path  = [tuple(x) for x in path]
      set_path = set(tuple(x) for x in path)
      unique_path = [x for x in path if x in set_path]

      paths_ent = all_paths[start_node]
      paths_ent.append(unique_path)
      all_paths[start_node] = paths_ent

  return all_paths



def get_X(all_paths, data, train_adj_dict):

  """ Dimension of embeddings = 2P, where P is the number of predicates
  """
  print('Generating vector representation ...')
  entities = list(train_adj_dict.keys())

  X_forward = np.zeros(shape=(data.nb_entities, data.nb_predicates))
  X_inverse = np.zeros(shape=(data.nb_entities, data.nb_predicates))

  for entity in entities:
    paths = all_paths[entity]
    for path in paths:

      if len(path)>0:

        forward = np.array(path)[:, 3]==1
        inverse = np.array(path)[:, 3]==-1

        # forward
        pred_idx = np.array(path)[forward,1]
        X_forward[entity, pred_idx] +=1

        # inverse
        pred_idx = np.array(path)[inverse,1]
        X_inverse[entity, pred_idx] +=1

      else:
        pass

  X = np.concatenate((X_forward, X_inverse), axis=1)
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
    all_paths = get_paths(train_adj_list, train_adj_dict, n_paths=n_paths, min_path_len=min_path_len, max_path_len=max_path_len)
    X = get_X(all_paths, data, train_adj_dict)
    np.savetxt(embeddings_path, X, fmt='%d')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
