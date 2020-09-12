#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys

import argparse

import multiprocessing
import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from kbc.training.batcher import Batcher

from kbc.models import DistMult, ComplEx
from kbc.regularizers import F2, N3
from kbc.evaluation import evaluate

from kbc.EM import EM
from kbc.EM import Data # we use an adapted Data class for EM

import logging
import os.path

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())

def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}\tH@50 {metrics["hits@50"]:.6f}\t' \
        f'H@100 {metrics["hits@100"]:.6f}'



## Clustering Utils
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

def cluster_and_make_triples(data, cluster_type, embeddings, n_clusters,
                             metric, eps=0.5, min_samples=5, damping=0.5):
  if cluster_type == 'kmeans':
      kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
      cluster_assignments = kmeans.predict(embeddings)

  if cluster_type == 'dbscan':
      allowed_metrics = ['precomputed', 'euclidean', 'cosine', 'chebyshev']
      if metric not in allowed_metrics:
          print('Invalid metric chosen!')
          metric = 'euclidean'
      clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(embeddings)
      cluster_assignments = clustering.labels_
      if -1 in cluster_assignments:
        cluster_assignments +=1

  if cluster_type == 'spectral':
      allowed_metrics = ['nearest_neighbors', 'rbf', 'precomputed']
      if metric not in allowed_metrics:
        print('Invalid metric chosen!')
        metric = 'nearest_neighbors' # choose an allowed metric
      clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", affinity = metric)
      cluster_assignments = clustering.fit_predict(embeddings)

  if cluster_type == 'affinity':
      allowed_metrics = ['euclidean', 'precomputed']
      if metric not in allowed_metrics:
          print('Invalid metric chosen!')
          metric = 'euclidean' # choose an allowed metric
      clustering = AffinityPropagation(damping=damping, affinity=metric)
      cluster_assignments = clustering.fit_predict(embeddings)
      if -1 in cluster_assignments:
          cluster_assignments +=1

  if cluster_type == 'random':
    # Generate random cluster assignments
    cluster_assignments = [np.random.randint(n_clusters) for ent in range(data.nb_entities)]

  # Covert assignment labels into lists of clusters
  print('Number of clusters we hoped to create: ' + str(n_clusters))
  print('Number of clusters retrieved: ' + str(len(np.unique(cluster_assignments))))

  n_clusters = len(np.unique(cluster_assignments))
  clusters = []
  for cluster_ID in np.unique(cluster_assignments):
    one_cluster = [i for i, x in enumerate(cluster_assignments) if x == cluster_ID]
    clusters.append(one_cluster)

  # New concept triples
  concept_triples = np.empty((0,3), int)
  for i, cluster in enumerate(clusters):
    o = ['concept__' + str(i) for k in range(len(cluster))]
    s = [data.idx_to_entity[ID] for ID in cluster]
    p = ['is_a_type_of' for i in range(len(cluster))]
    cluster_data = np.array([s, p, o]).transpose()
    cluster_data = cluster_data.reshape(len(cluster), 3)
    concept_triples = np.concatenate((concept_triples, cluster_data), axis=0)

  return concept_triples


def get_items(data):

  all_entities = list(data.entity_to_idx.keys())
  concepts = []
  for ent in all_entities:
    if ent[:9] == 'concept__':
      concepts += [ent]
  entities = list(np.setdiff1d(all_entities, concepts))
  isa_predicate = 'is_a_type_of'

  return entities, isa_predicate, concepts



def merge_and_export(train_path, alt_train_path, data_name, data, concept_triples):

  # Original dataset
  df = pd.read_csv(train_path, sep='\t', header = None, dtype=str)

  # pad with zeros to ensure labels match correctly
  if data_name == 'wn18rr':
    df[0] = df[0].apply(lambda x: '{0:0>8}'.format(x))
    df[2] = df[2].apply(lambda x: '{0:0>8}'.format(x))

  print('Length of original train dataset: ' + str(len(df)))
  print('Number of entities in the original dataset: ' +  str(data.nb_entities))
  print('Number of new triples: ' + str(concept_triples.shape[0]))

  # Concepts
  concept_df = pd.DataFrame({0: concept_triples[:, 0], 1: concept_triples[:, 1], 2: concept_triples[:, 2]})

  if data_name == 'wn18rr':
    concept_df[0] = concept_df[0].apply(lambda x: '{0:0>8}'.format(x))

  # Save new augmented dataset to tsv
  new_dataset = df.append(concept_df, ignore_index=True)
  new_dataset.to_csv(alt_train_path, sep ='\t', index = False, header = False)

  concept_triples_tuples = [tuple(t) for t in concept_df.to_numpy()]

  return concept_triples_tuples



def extract_concept_entities(data):
  '''This function takes the augmented training data and extracts all concept entities present.
  If we are doing sampling, it is possible that not all of the concept entities are added.'''
  all_entities = list(data.entity_to_idx.keys())
  concept_entities = []
  for ent in all_entities:
    if ent[:9] == 'concept__':
      concept_entities += [ent]

  return concept_entities




def get_ent_and_pred_indexes(data, device):

  # Entity embeddings
  concept_entities = extract_concept_entities(data)
  concept_entities_idx = []
  for concept_ent in concept_entities:
    concept_entities_idx.append(data.entity_to_idx[concept_ent])
  ent_index = np.setdiff1d(list(data.idx_to_entity.keys()), concept_entities_idx)

  tensor_ent_index = torch.tensor(ent_index, dtype=torch.long, device=device)

  # Predicate Embeddings
  concept_predicate = 'is_a_type_of'
  if concept_predicate in data.predicate_to_idx.keys():
    concept_predicate_idx = data.predicate_to_idx[concept_predicate]
  else:
    concept_predicate_idx = []
  pred_index = np.setdiff1d(list(data.idx_to_predicate.keys()), concept_predicate_idx)
  tensor_pred_index = torch.tensor(pred_index, dtype=torch.long, device=device)

  return tensor_ent_index, tensor_pred_index



### Sampling Utils
  def stratified_sampling(data, new_data, alpha=0, distribution='uniform'):
  # Setting alpha=0 results in no oversampling.

      print('Stratified sampling begins...')
      N = data.X.shape[0]
      C = (alpha * N) / (1 - alpha)
      nb_triples_to_sample = int(np.round(C, 0))

      nb_new_unique_triples = new_data.shape[0]
      new_triples_idx = [i for i in range(nb_new_unique_triples)]
      new_unique_triples = [tuple(new_data[i,:]) for i in range(nb_new_unique_triples)]

      # Uniform Distribution
      if distribution == 'uniform':
        p = [1 for i in range(nb_new_unique_triples)]
        p = p/np.sum(p)

      # Pick triples to add to the dataset
      indicies = np.random.choice(new_triples_idx, nb_triples_to_sample, p=p)
      sampled_data = new_data[indicies, :]

      return sampled_data




def main(argv):
    parser = argparse.ArgumentParser('Meta-KBC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='complex',
                        choices=['distmult', 'complex'])

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)
    parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--F2', action='store', type=float, default=None)
    parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate-every', '-V', action='store', type=int, default=None)

    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    parser.add_argument('--quiet', '-q', action='store_true', default=True)

    parser.add_argument('--data_name', action='store', type=str, required=True, default=None)

    # Destination of performance metrics
    parser.add_argument('--results_csv', action='store', type=str, default=None)

    # load entity representations
    parser.add_argument('--embeddings_path', action='store', type=str, default=None)

    # Clustering
    parser.add_argument('--n_clusters', action='store', type=int, default=None)
    parser.add_argument('--cluster_type', action='store', type=str, default='kmeans',
                        choices=['dbscan', 'spectral', 'affinity', 'kmeans', 'random'])

    # Parameters for Clustering
    parser.add_argument('--eps', action='store', type=float, default=0.5)
    parser.add_argument('--min_samples', action='store', type=int, default=5)
    parser.add_argument('--metric', action='store', type=str, default='NA',
                    choices=['precomputed', 'euclidean', 'cosine', 'chebyshev', 'nearest_neighbors', 'rbf', 'NA'])
    parser.add_argument('--damping', action = 'store', type=float, default=0.5)

    # Destination of new train .tsv
    parser.add_argument('--alt_train', action='store', type=str)

    # Optional stratified sampling
    parser.add_argument('--alpha', action='store', type=float, default=None)

    #EM param
    parser.add_argument('--run_EM', action = 'store', type=float, default=True)
    parser.add_argument('--E_every', action = 'store', type=float, default=1) # how many epochs between E steps
    parser.add_argument('--em_setting', action = 'store', type=str, default='deterministic') # E-step selection


    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))


    data_name = args.data_name
    eval_path = args.train
    results_csv_path = args.results_csv

    embeddings_path = args.embeddings_path
    alpha = args.alpha
    alt_train_path = args.alt_train

    metric = args.metric
    damping = args.damping
    n_clusters = args.n_clusters
    cluster_type = args.cluster_type
    eps = args.eps
    min_samples = args.min_samples

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    test_i_path = args.test_i
    test_ii_path = args.test_ii

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size

    batch_size = args.batch_size
    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size

    nb_epochs = args.epochs

    seed = args.seed
    learning_rate = args.learning_rate

    F2_weight = args.F2
    N3_weight = args.N3

    validate_every = args.validate_every
    input_type = args.input_type

    load_path = args.load
    save_path = args.save

    is_quiet = args.quiet

    run_EM = args.run_EM
    E_every = args.E_every
    em_setting = args.em_setting

    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # activate GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')


    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
              test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)


    print('Loading embeddings...')
    embeddings = np.loadtxt(embeddings_path, dtype=int)

    print('Clustering...')
    concept_triples = cluster_and_make_triples(data, cluster_type, embeddings, n_clusters,
                            metric, eps=eps, min_samples=min_samples, damping=damping)

    if alpha is not None:
      print('Sampling...')
      concept_triples = stratified_sampling(data, concept_triples, alpha=alpha)

    print('Exporting new dataset...')
    concept_triples_tuples = merge_and_export(train_path, alt_train_path, data_name, data, concept_triples)

    print('Training ComplEx...')
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data = Data(train_path=alt_train_path, dev_path=dev_path, test_path=test_path,
              test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
      (data.dev_triples, 'dev'),
      (data.test_triples, 'test'),
      (data.test_i_triples, 'test-I'),
      (data.test_ii_triples, 'test-II'),
    ]


    if run_EM:
      print('Initialising EM...')
      em = EM(concept_triples_tuples, data, device)


    # For evaluation and metrics
    tensor_ent_index, tensor_pred_index = get_ent_and_pred_indexes(data, device)

    original_data = Data(train_path=eval_path, dev_path=dev_path, test_path=test_path,
          test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    original_triples_name_pairs = [
      (original_data.dev_triples, 'dev'),
      (original_data.test_triples, 'test'),
      (original_data.test_i_triples, 'test-I'),
      (original_data.test_ii_triples, 'test-II'),
    ]



    # Define model
    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=True)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=True)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    param_module = nn.ModuleDict({'entities': entity_embeddings, 'predicates': predicate_embeddings}).to(device)
    if load_path is not None:
      param_module.load_state_dict(torch.load(load_path))

    parameter_lst = nn.ParameterList([entity_embeddings.weight, predicate_embeddings.weight])

    model_factory = {
      'distmult': lambda: DistMult(entity_embeddings=entity_embeddings),
      'complex': lambda: ComplEx(entity_embeddings=entity_embeddings)
    }

    assert model_name in model_factory
    model = model_factory[model_name]()
    model.to(device)

    print('Model state:')
    for param_tensor in param_module.state_dict():
      print(f'\t{param_tensor}\t{param_module.state_dict()[param_tensor].size()}')

    optimizer_factory = {
      'adagrad': lambda: optim.Adagrad(parameter_lst, lr=learning_rate),
      'adam': lambda: optim.Adam(parameter_lst, lr=learning_rate),
      'sgd': lambda: optim.SGD(parameter_lst, lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    F2_reg = N3_reg = None

    if F2_weight is not None:
      F2_reg = F2()
    if N3_weight is not None:
      N3_reg = N3()

    # train embeddings
    for epoch_no in range(1, nb_epochs + 1):
      batcher = Batcher(data.nb_examples, batch_size, 1, random_state)
      nb_batches = len(batcher.batches)

      epoch_loss_values = []
      for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
          indices = batcher.get_batch(batch_start, batch_end)
          x_batch = torch.tensor(data.X[indices, :], dtype=torch.long, device=device)

          xs_batch_emb = entity_embeddings(x_batch[:, 0])
          xp_batch_emb = predicate_embeddings(x_batch[:, 1])
          xo_batch_emb = entity_embeddings(x_batch[:, 2])

          sp_scores, po_scores = model.forward(xp_batch_emb, xs_batch_emb, xo_batch_emb)
          factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

          s_loss = loss_function(sp_scores, x_batch[:, 2])
          o_loss = loss_function(po_scores, x_batch[:, 0])

          loss = s_loss + o_loss

          if F2_weight is not None:
              loss += F2_weight * F2_reg(factors)

          if N3_weight is not None:
              loss += N3_weight * N3_reg(factors)

          loss.backward()

          optimizer.step()
          optimizer.zero_grad()

          loss_value = loss.item()
          epoch_loss_values += [loss_value]

          if not is_quiet:
              print(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

      loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
      print(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')



      ### Initalise metrics
      if epoch_no == 1:
          entity_embeddings_no_clusters = entity_embeddings(tensor_ent_index)
          entity_embeddings_nc = nn.Embedding(original_data.nb_entities, rank, sparse=True)
          entity_embeddings_nc.weight.data = entity_embeddings_no_clusters

          predicate_embeddings_no_clusters = predicate_embeddings(tensor_pred_index)
          predicate_embeddings_nc = nn.Embedding(original_data.nb_predicates, rank, sparse=True)
          predicate_embeddings_nc.weight.data = predicate_embeddings_no_clusters

          for triples, name in [(t, n) for t, n in original_triples_name_pairs if len(t) > 0]:
              metrics = evaluate(entity_embeddings=entity_embeddings_nc, predicate_embeddings=predicate_embeddings_nc,
                                  test_triples=triples, all_triples=original_data.all_triples,
                                  entity_to_index=original_data.entity_to_idx, predicate_to_index=original_data.predicate_to_idx,
                                  model=model, batch_size=eval_batch_size, device=device)
              if name == 'dev':
                  best_metrics_dev = metrics

              if name == 'test':
                  best_MRR = metrics['MRR']
                  best_metrics_test = metrics
                  best_epoch = epoch_no


              print(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')


      ### Compute metrics every V epochs
      if validate_every is not None and epoch_no % validate_every == 0 and epoch_no >1:

          entity_embeddings_no_clusters = entity_embeddings(tensor_ent_index)
          entity_embeddings_nc = nn.Embedding(original_data.nb_entities, rank, sparse=True)
          entity_embeddings_nc.weight.data = entity_embeddings_no_clusters

          predicate_embeddings_no_clusters = predicate_embeddings(tensor_pred_index)
          predicate_embeddings_nc = nn.Embedding(original_data.nb_predicates, rank, sparse=True)
          predicate_embeddings_nc.weight.data = predicate_embeddings_no_clusters

          for triples, name in [(t, n) for t, n in original_triples_name_pairs if len(t) > 0]:
              metrics = evaluate(entity_embeddings=entity_embeddings_nc, predicate_embeddings=predicate_embeddings_nc,
                                  test_triples=triples, all_triples=original_data.all_triples,
                                  entity_to_index=original_data.entity_to_idx, predicate_to_index=original_data.predicate_to_idx,
                                  model=model, batch_size=eval_batch_size, device=device)
              if name == 'dev':
                  if metrics['MRR'] > best_MRR:
                      best_epoch = epoch_no
                      best_MRR = metrics['MRR']
                      best_metrics_dev = metrics
                      update = 'yes'
                  else:
                      update = 'no'

              if (name == 'test' and update =='yes') :
                  best_metrics_test = metrics


              print(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')


      if run_EM and epoch_no % E_every==0:
        print('E-step...')
        new_train = em.E_step(model, original_data, entity_embeddings, predicate_embeddings, setting = em_setting)
        data.augmenttrain(new_train)



    # End of training results
    entity_embeddings_no_clusters = entity_embeddings(tensor_ent_index)
    entity_embeddings_nc = nn.Embedding(original_data.nb_entities, rank, sparse=True)
    entity_embeddings_nc.weight.data = entity_embeddings_no_clusters

    predicate_embeddings_no_clusters = predicate_embeddings(tensor_pred_index)
    predicate_embeddings_nc = nn.Embedding(original_data.nb_predicates, rank, sparse=True)
    predicate_embeddings_nc.weight.data = predicate_embeddings_no_clusters

    for triples, name in [(t, n) for t, n in original_triples_name_pairs if len(t) > 0]:
      metrics = evaluate(entity_embeddings=entity_embeddings_nc, predicate_embeddings=predicate_embeddings_nc,
                          test_triples=triples, all_triples=original_data.all_triples,
                          entity_to_index=original_data.entity_to_idx, predicate_to_index=original_data.predicate_to_idx,
                          model=model, batch_size=eval_batch_size, device=device)

      print(f'Final \t{name} results\t{metrics_to_str(metrics)}')


        ### Append best results to csv
    for metrics, name in [[best_metrics_dev, 'dev'], [best_metrics_test, 'test']]:
      results_original = pd.DataFrame({'dataset': data_name, 'dev/test': name, 'em-setting': em_setting, 'n_clusters_init' : n_clusters , 'n_clusters_end':em.nb_current_concepts, 'cluster_type': cluster_type, 'metric': metric, 'embeddings_string' : embeddings_path,  'MRR': metrics['MRR'] , 'H1': metrics['hits@1'], 'H3':  metrics['hits@3'], 'H5': metrics['hits@5'], 'H10': metrics['hits@10'], 'H50': metrics['hits@50'], 'H100': metrics['hits@100'], 'best_epoch': best_epoch,'lr': learning_rate, 'F2': F2_weight, 'N3': N3_weight, 'seed' : seed, 'model':model_name,  'embedding_size': embedding_size, 'batch_size': batch_size, 'nb_epochs':nb_epochs}, index=[0])

      if os.path.exists(results_csv_path):
          all_results = pd.read_csv(results_csv_path)
          all_results = all_results.append(results_original, ignore_index=True)
          all_results.to_csv(results_csv_path, index=False, header=True)
          print('Appended! Destination: {}'.format(results_csv_path))
      else:
          results_original.to_csv(results_csv_path, index=False, header=True)
          print('Appended!')

    if save_path is not None:
        torch.save(param_module.state_dict(), save_path)

    logger.info("Training Finished")

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
