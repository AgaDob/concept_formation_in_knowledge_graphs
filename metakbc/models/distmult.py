# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from metakbc.models.base import BaseModel

from typing import Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class DistMult(BaseModel):
    def __init__(self,
                 entity_embeddings: Optional[nn.Embedding] = None) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        # [B]
        res = torch.sum(rel * arg1 * arg2, 1)
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                entity_embeddings: Optional[Tensor] = None,
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        assert (entity_embeddings is not None) or (self.entity_embeddings is not None)

        # [N, E]
        emb = self.entity_embeddings.weight if entity_embeddings is None else entity_embeddings

        score_sp = score_po = None

        if arg1 is not None:
            # [B, N] = [B, E] @ [E, N]
            score_sp = (rel * arg1) @ emb.t()

        if arg2 is not None:
            # [B, N] = [B, E] @ [E, N]
            score_po = (rel * arg2) @ emb.t()

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor,
               safe: bool = False) -> Tensor:
        return embedding_vector
