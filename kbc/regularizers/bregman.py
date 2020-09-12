# -*- coding: utf-8 -*-

from kbc.regularizers import Regularizer, AdaptiveRegularizer

import torch
from torch import nn, Tensor

from typing import Dict


class DiagonalMahalanobisAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 factor_size: int):
        super().__init__(regularizer)
        self.factor_size = factor_size
        init = torch.zeros(self.factor_size, dtype=torch.float32)
        self.A = nn.Parameter(init, requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([factor * self.A], dim=1)
        return torch.sum(norm_values)

    def project_(self):
        self.A.data.clamp_(0.0)

    def values_(self) -> Dict[str, float]:
        ni = self.A.shape[0]
        tmp = self.A.data.cpu().numpy()[:].tolist()
        return {f'A{i}': tmp[i] for i in range(ni)}


class ProjectedDiagonalMahalanobisAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 factor_size: int):
        super().__init__(regularizer)
        self.factor_size = factor_size
        self.projection = nn.Linear(factor_size, factor_size)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([self.projection(factor)], dim=1)
        return torch.sum(norm_values)

    def project_(self):
        self.projection.weight.data.clamp_(0.0)

    def values_(self) -> Dict[str, float]:
        return {}
