# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from typing import Tuple, List, Dict, Set, Optional

import logging

logger = logging.getLogger(__name__)


def to_networkx(triples: List[Tuple[str, str, str]],
                entity_to_idx: Dict[str, int],
                predicate_to_idx: Dict[str, int],
                predicates: Optional[Set[str]] = None,
                is_multidigraph: bool = False) -> nx.DiGraph:

    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]

    G = nx.MultiDiGraph() if is_multidigraph else nx.DiGraph()

    entities = sorted({s for (s, _, _) in triples} | {o for (_, _, o) in triples})
    G.add_nodes_from([entity_to_idx[e] for e in entities])

    if is_multidigraph:
        G.add_edges_from([(entity_to_idx[s], entity_to_idx[o], {'p': predicate_to_idx[p]}) for s, p, o in _triples])
    else:
        edge_lst = sorted({(entity_to_idx[s], entity_to_idx[o]) for s, p, o in _triples})
        G.add_edges_from(edge_lst)

    return G


def get_relational_features(triples: List[Tuple[str, str, str]],
                            entity_to_idx: Dict[str, int],
                            is_bool: bool = False,
                            predicates: Optional[Set[str]] = None,
                            is_in: bool = True,
                            is_out: bool = True,
                            log_scale: bool = False) -> np.ndarray:
    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]
    _predicates = sorted({p for _, p, _ in _triples})

    _predicate_to_idx = {p: i for i, p in enumerate(_predicates)}

    nb_entities = max(v for _, v in entity_to_idx.items()) + 1
    nb_features = max(v for _, v in _predicate_to_idx.items()) + 1

    res_in = np.zeros(shape=(nb_entities, nb_features), dtype=np.float32) if is_in is True else None
    res_out = np.zeros(shape=(nb_entities, nb_features), dtype=np.float32) if is_out is True else None

    for s, p, o in _triples:
        s_idx, o_idx = entity_to_idx[s], entity_to_idx[o]
        p_idx = _predicate_to_idx[p]

        if is_bool is True:
            if res_in is not None:
                res_in[o_idx, p_idx] = 1
            if res_out is not None:
                res_out[s_idx, p_idx] = 1
        else:
            if res_in is not None:
                res_in[o_idx, p_idx] += 1
            if res_out is not None:
                res_out[s_idx, p_idx] += 1

    tmp_lst = [x for x in [res_in, res_out] if x is not None]

    assert len(tmp_lst) > 0

    res = tmp_lst[0] if len(tmp_lst) == 1 else np.concatenate(tmp_lst, axis=1)

    if log_scale is True:
        res = np.log(res)

    return res


# @profile
def get_graph_features(triples: List[Tuple[str, str, str]],
                       entity_to_idx: Dict[str, int],
                       predicate_to_idx: Dict[str, int],
                       predicates: Optional[Set[str]] = None,
                       log_scale: bool = False) -> np.ndarray:
    G = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=False)
    # uG = G.to_undirected()

    mG = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=True)
    # umG = mG.to_undirected()

    logger.debug('mG.degree() ..')
    f1 = mG.degree()

    logger.debug('mG.in_degree() ..')
    f2 = mG.in_degree()

    logger.debug('mG.out_degree() ..')
    f3 = mG.out_degree()

    logger.debug('nx.pagerank(G) ..')
    f4 = nx.pagerank(G)

    logger.debug('nx.degree_centrality(mG) ..')
    f5 = nx.degree_centrality(mG)

    logger.debug('nx.in_degree_centrality(mG) ..')
    f6 = nx.in_degree_centrality(mG)

    logger.debug('nx.out_degree_centrality(mG) ..')
    f7 = nx.out_degree_centrality(mG)

    def to_log_scale(entries):
        if log_scale is False:
            res = entries
        else:
            res = []
            for k, v in (entries.items() if isinstance(entries, dict) else entries):
                res += [(k, np.log(1 + v))]
        return res

    feature_lst = [to_log_scale(f1), to_log_scale(f2), to_log_scale(f3), f4, f5, f6, f7]

    nb_entities = max(v for _, v in entity_to_idx.items()) + 1
    nb_features = len(feature_lst)

    res = np.zeros(shape=(nb_entities, nb_features), dtype=np.float32)

    for i, f in enumerate(feature_lst):
        for k, v in (f.items() if isinstance(f, dict) else f):
            res[k, i] = v

    return res


if __name__ == '__main__':
    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'c'),
        ('b', 'q', 'd')
    ]

    entity_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    predicate_to_idx = {'p': 0, 'q': 1}

    features = get_graph_features(triples, entity_to_idx, predicate_to_idx)
    print(features)

    features = get_relational_features(triples, entity_to_idx)
    print(features)
