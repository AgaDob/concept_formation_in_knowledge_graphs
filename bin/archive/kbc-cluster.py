#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim

from metakbc.training.data import Data
from metakbc.training.batcher import Batcher

from metakbc.models import DistMult, ComplEx

from metakbc.regularizers import F2, N3
from metakbc.evaluation import evaluate

import logging

import pandas as pd
import os.path

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}\tH@50 {metrics["hits@50"]:.6f}\t' \
        f'H@100 {metrics["hits@100"]:.6f}'


def main(argv):
    parser = argparse.ArgumentParser('Meta-KBC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='distmult',
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

    # new
    parser.add_argument('--n_clusters', action='store', type=int, default=None)
    parser.add_argument('--data_name', action='store', type=str, default=None)
    parser.add_argument('--alt_train', action='store', type=str)
    parser.add_argument('--results_csv', action='store', type=str, default=None)

    parser.add_argument('--learning-rate_c', action='store', type=float, default=0.1)
    parser.add_argument('--F2_c', action='store', type=float, default=None)
    parser.add_argument('--N3_c', action='store', type=float, default=None)

    args = parser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    n_clusters = args.n_clusters
    data_name = args.data_name
    alt_train_path = args.alt_train
    results_csv_path = args.results_csv

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

    if args.seed == 0:
        seed = args.seed
    else:
        seed = np.random.randint(100000)

    learning_rate = args.learning_rate

    F2_weight = args.F2
    N3_weight = args.N3

    validate_every = args.validate_every
    input_type = args.input_type

    load_path = args.load
    save_path = args.save

    is_quiet = args.quiet

    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

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

    logger.info('Model state:')
    for param_tensor in param_module.state_dict():
        logger.info(f'\t{param_tensor}\t{param_module.state_dict()[param_tensor].size()}')

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
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

        if validate_every is not None and epoch_no % validate_every == 0:
            for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
                metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                   test_triples=triples, all_triples=data.all_triples,
                                   entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                   model=model, batch_size=eval_batch_size, device=device)
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')

    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

        ## Append results to csv
        if results_csv_path:

            results_original = pd.DataFrame({'dataset': data_name, 'dev/test': name, 'n_clusters' : 0 , 'MRR': metrics['MRR'] , 'H1': metrics['hits@1'], 'H3':  metrics['hits@3'], 'H5': metrics['hits@5'], 'H10': metrics['hits@10'], 'H50': metrics['hits@50'], 'H100': metrics['hits@100'], 'lr': learning_rate, 'F2': F2_weight, 'N3': N3_weight, 'seed' : seed, 'model':model_name,  'embedding_size': embedding_size, 'batch_size': batch_size, 'nb_epochs':nb_epochs}, index=[0])
            # check if the csv already exists
            if os.path.exists(results_csv_path):
                # load results csv into df
                all_results = pd.read_csv(results_csv_path)
                # append to it
                all_results = all_results.append(results_original, ignore_index=True)
                # export it
                all_results.to_csv(results_csv_path, index=False, header=True)
            else:
                results_original.to_csv(results_csv_path, index=False, header=True)

    if save_path is not None:
        torch.save(param_module.state_dict(), save_path)

    logger.info("1st Training finished")


    ## Cluster Embeddings
    from sklearn.cluster import KMeans

    def get_embeddings(data, model):
      """ Get Embeddings and Embeddings Dict
      """

      entities = [i for i in range(data.nb_entities)]
      embeddings = model.entity_embeddings(torch.LongTensor(entities).cuda())
      embeddings = torch.Tensor.cpu(embeddings)
      embeddings = embeddings.detach().numpy()

      embeddings_list = [embeddings[i,:] for i in range(embeddings.shape[0])]
      embeddings_dict = {key:val for key, val in zip(entities, embeddings_list)}

      return embeddings, embeddings_dict

    embeddings, embeddings_dict = get_embeddings(data, model)
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
    cluster_assignments = kmeans.predict(embeddings)

    clusters = []
    for cluster_ID in range(n_clusters):
      one_cluster = [i for i, x in enumerate(cluster_assignments) if x == cluster_ID]
      clusters.append(one_cluster)


    ###### 2. Add clusters to the dataset

    # original dataset
    df = pd.read_csv(args.train, sep='\t', header = None)

    # new triples
    new_data = np.empty((0,3), int)
    for i, cluster in enumerate(clusters):
      o = ['concept__' + str(i) for k in range(len(cluster))]
      s = [data.idx_to_entity[ID] for ID in cluster]
      p = ['is_a_type_of' for i in range(len(cluster))]
      cluster_data = np.array([s, p, o]).transpose()
      cluster_data = cluster_data.reshape(len(cluster), 3)
      new_data = np.concatenate((new_data, cluster_data), axis=0)

    new_df = pd.DataFrame({0: new_data[:, 0], 1: new_data[:, 1], 2: new_data[:, 2]})

    # combine and export
    new_dataset = df.append(new_df)
    new_dataset.to_csv(alt_train_path, sep ='\t', index = False, header = False)



    ###### 3. Train new embeddings
    learning_rate = args.learning_rate_c
    F2_weight = args.F2_c
    N3_weight = args.N3_c

    logger.info('Training Augmented Embeddings')
    data = Data(train_path=alt_train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

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

    logger.info('Model state:')
    for param_tensor in param_module.state_dict():
        logger.info(f'\t{param_tensor}\t{param_module.state_dict()[param_tensor].size()}')

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
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

        if validate_every is not None and epoch_no % validate_every == 0:
            for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
                metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                   test_triples=triples, all_triples=data.all_triples,
                                   entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                   model=model, batch_size=eval_batch_size, device=device)
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')


    ####### Evaluate on a dataset without cluster entities #####
    original_data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
            test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    triples_name_pairs = [
        (original_data.dev_triples, 'dev'),
        (original_data.test_triples, 'test'),
        (original_data.test_i_triples, 'test-I'),
        (original_data.test_ii_triples, 'test-II'),
    ]

    # Discard Embedding for Concepts and Concept Relation

    # Entity embeddings
    concept_entities = ['concept__' + str(i) for i in range(n_clusters)]
    concept_entities_idx = []
    for concept_ent in concept_entities:
      concept_entities_idx.append(data.entity_to_idx[concept_ent])
    ent_index = np.setdiff1d(list(data.idx_to_entity.keys()), concept_entities_idx)

    tensor_ent_index = torch.tensor(ent_index, dtype=torch.long, device=device)
    entity_embeddings_no_clusters = entity_embeddings(tensor_ent_index)

    entity_embeddings_nc = nn.Embedding(original_data.nb_entities, rank, sparse=True)
    entity_embeddings_nc.weight.data = entity_embeddings_no_clusters

    # Predicate Embeddings
    concept_predicate = 'is_a_type_of'
    concept_predicate_idx = data.predicate_to_idx[concept_predicate]
    pred_index = np.setdiff1d(list(data.idx_to_predicate.keys()), concept_predicate_idx)

    tensor_pred_index = torch.tensor(pred_index, dtype=torch.long, device=device)
    predicate_embeddings_no_clusters = predicate_embeddings(tensor_pred_index)

    predicate_embeddings_nc = nn.Embedding(original_data.nb_predicates, rank, sparse=True)
    predicate_embeddings_nc.weight.data = predicate_embeddings_no_clusters

    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=entity_embeddings_nc, predicate_embeddings=predicate_embeddings_nc,
                            test_triples=triples, all_triples=original_data.all_triples,
                            entity_to_index=original_data.entity_to_idx, predicate_to_index=original_data.predicate_to_idx,
                            model=model, batch_size=eval_batch_size, device=device)

        logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

        ## Append results to csv
        if results_csv_path:

            results_original = pd.DataFrame({'dataset': data_name, 'dev/test': name, 'n_clusters' : n_clusters , 'MRR': metrics['MRR'] , 'H1': metrics['hits@1'], 'H3':  metrics['hits@3'], 'H5': metrics['hits@5'], 'H10': metrics['hits@10'], 'H50': metrics['hits@50'], 'H100': metrics['hits@100'], 'lr': learning_rate, 'F2': F2_weight, 'N3': N3_weight, 'seed' : seed, 'model':model_name,  'embedding_size': embedding_size, 'batch_size': batch_size, 'nb_epochs':nb_epochs}, index=[0])
            # check if the csv already exists
            if os.path.exists(results_csv_path):
                print(os.path.exists(results_csv_path))
                # load results csv into df
                all_results = pd.read_csv(results_csv_path)
                # append to it
                all_results = all_results.append(results_original, ignore_index=True)
                # export it
                all_results.to_csv(results_csv_path, index=False, header=True)
            else:
                results_original.to_csv(results_csv_path, index=False, header=True)


    if save_path is not None:
        torch.save(param_module.state_dict(), save_path)
    logger.info("2nd Training finished")







if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])