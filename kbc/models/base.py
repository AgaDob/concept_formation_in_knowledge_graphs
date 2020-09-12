# -*- coding: utf-8 -*-

from torch import nn, Tensor

from abc import ABC, abstractmethod

from typing import Tuple, Optional, List

import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def score(self,
              rel: Tensor,
              arg1: Tensor,
              arg2: Tensor,
              *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                *args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def factor(self,
               embedding_vector: Tensor,
               safe: bool = False) -> Tensor:
        raise NotImplementedError
