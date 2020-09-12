# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn, Tensor

from kbc.models.base import BaseModel

from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ComplEx(BaseModel):
    def __init__(self,
                 entity_embeddings: Optional[nn.Embedding] = None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.entity_embeddings = entity_embeddings

        eps = torch.tensor(1e-45, dtype=torch.float32, requires_grad=False)
        self.eps = eps.to(device) if device is not None else eps

    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        rank = rel.shape[1] // 2

        # [B, E]
        rel_real, rel_img = rel[:, :rank], rel[:, rank:]
        arg1_real, arg1_img = arg1[:, :rank], arg1[:, rank:]
        arg2_real, arg2_img = arg2[:, :rank], arg2[:, rank:]

        # [B] Tensor
        score1 = torch.sum(rel_real * arg1_real * arg2_real, 1)
        score2 = torch.sum(rel_real * arg1_img * arg2_img, 1)
        score3 = torch.sum(rel_img * arg1_real * arg2_img, 1)
        score4 = torch.sum(rel_img * arg1_img * arg2_real, 1)

        res = score1 + score2 + score3 - score4

        # [B] Tensor
        return res

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                entity_embeddings: Optional[Tensor] = None,
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        rank = rel.shape[1] // 2

        assert (entity_embeddings is not None) or (self.entity_embeddings is not None)

        emb = self.entity_embeddings.weight if entity_embeddings is None else entity_embeddings

        rel_real, rel_img = rel[:, :rank], rel[:, rank:]
        emb_real, emb_img = emb[:, :rank], emb[:, rank:]

        # [B] Tensor
        score_sp = score_po = None

        if arg1 is not None:
            arg1_real, arg1_img = arg1[:, :rank], arg1[:, rank:]

            score1_sp = (rel_real * arg1_real) @ emb_real.t()
            score2_sp = (rel_real * arg1_img) @ emb_img.t()
            score3_sp = (rel_img * arg1_real) @ emb_img.t()
            score4_sp = (rel_img * arg1_img) @ emb_real.t()

            score_sp = score1_sp + score2_sp + score3_sp - score4_sp

        if arg2 is not None:
            arg2_real, arg2_img = arg2[:, :rank], arg2[:, rank:]

            score1_po = (rel_real * arg2_real) @ emb_real.t()
            score2_po = (rel_real * arg2_img) @ emb_img.t()
            score3_po = (rel_img * arg2_img) @ emb_real.t()
            score4_po = (rel_img * arg2_real) @ emb_img.t()

            score_po = score1_po + score2_po + score3_po - score4_po

        return score_sp, score_po

    def factor(self,
               embedding_vector: Tensor,
               safe: bool = False) -> Tensor:
        rank = embedding_vector.shape[1] // 2

        vec_real = embedding_vector[:, :rank]
        vec_img = embedding_vector[:, rank:]

        sq_factor = vec_real ** 2 + vec_img ** 2
        if safe is True:
            sq_factor = torch.max(sq_factor, self.eps)

        return torch.sqrt(sq_factor)
