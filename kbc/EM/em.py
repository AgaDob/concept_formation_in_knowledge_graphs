# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from typing import Optional, List, Tuple, Dict

import argparse

import multiprocessing
import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from kbc.training.data import Data
from kbc.training.batcher import Batcher

from kbc.models import DistMult, ComplEx
from kbc.regularizers import F2, N3
from kbc.evaluation import evaluate

import logging
import os.path



def get_items(data):
  all_entities = list(data.entity_to_idx.keys())
  concepts = []
  for ent in all_entities:
    if ent[:9] == 'concept__':
      concepts += [ent]
  entities = list(np.setdiff1d(all_entities, concepts))
  isa_predicate = 'is_a_type_of'

  return entities, isa_predicate, concepts


class EM():
  def __init__(self, init_concept_triples, data, device):
    self.entities, self.isa_predicate, self.concepts = get_items(data)
    self.n_clusters = len(self.concepts)
    self.n_iters = 0
    self.concept_triples = []
    if init_concept_triples is not None:
      self.concept_triples += [init_concept_triples]
    self.scores = []
    self.entity_to_index = data.entity_to_idx
    self.predicate_to_index = data.predicate_to_idx
    self.device = device

    self.d = {}
    for entity in self.entities:
      arr = np.array([(entity, self.isa_predicate, concept) for concept in self.concepts])
      xs = np.array([self.entity_to_index.get(s) for (s, _, _) in arr])
      xp = np.array([self.predicate_to_index.get(p) for (_, p, _) in arr])
      xo = np.array([self.entity_to_index.get(o) for (_, _, o) in arr])

      with torch.no_grad():
        tensor_xs = torch.tensor(xs, dtype=torch.long, device=self.device)
        tensor_xp = torch.tensor(xp, dtype=torch.long, device=self.device)
        tensor_xo = torch.tensor(xo, dtype=torch.long, device=self.device)

      self.d[entity] = (tensor_xs, tensor_xp, tensor_xo)


  def E_step(self, model, original_data,  entity_embeddings, predicate_embeddings, setting='deterministic'):
    self.n_iters += 1
    self.setting = setting
    new_concepts = []
    scores_l = []

    for entity in self.entities:
      tensor_xs, tensor_xp, tensor_xo = self.d[entity]

      tensor_xs_emb = entity_embeddings(tensor_xs)
      tensor_xp_emb = predicate_embeddings(tensor_xp)
      tensor_xo_emb = entity_embeddings(tensor_xo)

      scores = model.score(tensor_xp_emb, tensor_xs_emb, tensor_xo_emb).detach().cpu().numpy()

      if self.setting == 'deterministic':
        new_concept = np.argmax(scores)

      elif self.setting == 'stochastic':
        # shift to avoid negative and 0 probabilities
        scores_positive = scores + abs(np.min(scores)) + 0.0000001
        # normalise
        p = scores_positive/np.sum(scores_positive)
        # sample
        new_concept = np.random.choice([i for i in range(self.n_clusters)], 1, p=p)[0]


      new_concepts += [self.concepts[new_concept]]
      scores_l += [scores]

    # create new train dataset
    new_triples = [(entity, self.isa_predicate, new_concept) for entity, new_concept in zip(self.entities, new_concepts)]
    new_train = original_data.train_triples + new_triples

    # save for analysis
    self.concept_triples += [new_triples]
    self.scores += [scores_l]
    self._concept_statistics()

    return new_train

  def _concept_statistics(self):
    current = self.concept_triples[-1]
    previous = self.concept_triples[-2]

    current_set = set(current)
    previous_set = set(previous)
    differences = (previous_set - current_set)
    print('Number of assignment changes: {} out of {}.  ({}%)'.format(len(differences), len(current),
                                                                      np.round(len(differences)/len(current)*100), 0))

    self.nb_current_concepts = len(set(np.array(current)[:, 2]))
    print('Current number of concepts: {}'.format(self.nb_current_concepts))
