# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from kbc.models.base import BaseModel

from typing import Tuple, Optional

import logging

logger = logging.getLogger(__name__)


class Bilinear(BaseModel):
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
        return torch.einsum('bx,bxy,by->b', arg1, rel, arg2)

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
            # [B E] [B E E] [N E] -> [B N]
            score_sp = torch.einsum('bx,bxy,ny->bn', arg1, rel, emb)

        if arg2 is not None:
            # [N E] [B E E] [B E] -> [B N]
            score_po = torch.einsum('nx,bxy,by->bn', emb, rel, arg2)

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor,
               safe: bool = False) -> Tensor:
        return embedding_vector
