#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# This script:
# 1) Represents each entity using a vector of length 2P, where P is the number of predicates.
# Either counting or binary indication for when given entitity participates as 1) a subject or 2) an object with the given relation.
# 2) Saves to embeddings matrix



import os
import sys
import argparse
import logging
import os.path

import multiprocessing
import numpy as np
import pandas as pd

from metakbc.training.data import Data


def propositionalization_2P(data, binary= False):

  # initate representations
  X_subjects = np.zeros(shape=(data.nb_entities, data.nb_predicates))
  X_objects = np.zeros(shape=(data.nb_entities, data.nb_predicates))

  all_triples = data.X

  for i in range(data.X.shape[0]):
    triple = list(all_triples[i, :])
    s, p, o = triple

    if not binary:
      # add all occurances
      X_subjects[s,p] += 1
      X_objects[o,p] += 1

    else:
      # use binary indicator
      X_subjects[s,p] = 1
      X_objects[o,p] = 1

  X = np.concatenate((X_subjects, X_objects), axis = 1)

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
    # Count occurances or use binary representations
    parser.add_argument('--binary', action='store', type=bool, default=False)


    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    embeddings_path = args.embeddings_path
    data_name = args.data_name
    binary = args.binary

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    test_i_path = args.test_i
    test_ii_path = args.test_ii
    input_type = args.input_type


    print('creating dataset...')
    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    print('generating matrix...')
    X = propositionalization_2P(data, binary=binary)

    print('saving embeddings...')
    np.savetxt(embeddings_path, X, fmt='%d')



if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
