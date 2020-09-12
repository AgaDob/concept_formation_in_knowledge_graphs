# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from kbc.regularizers import Regularizer
from kbc.regularizers import NX

from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)


class AdaptiveRegularizer(nn.Module, ABC):
    def __init__(self,
                 regularizer: Regularizer) -> None:
        super().__init__()
        self.regularizer = regularizer

    @abstractmethod
    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def project_(self):
        raise NotImplementedError

    @abstractmethod
    def values_(self) -> Dict[str, int]:
        raise NotImplementedError


class NXAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self, p: float = 3.0):
        super().__init__(NX(nn.Parameter(torch.tensor(p, dtype=torch.float32), requires_grad=True)))
        self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([factor], dim=1)
        return self.weight * torch.sum(norm_values)

    def project_(self):
        self.regularizer.p.data.clamp_(0.0)
        self.weight.data.clamp_(0.0)

    def values_(self) -> Dict[str, float]:
        return {'weight': self.weight.item(), 'p': self.regularizer.p.item()}


class FixedLambdaNXAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 p: float = 3.0,
                 lambda_value: float = 0.1,
                 device: Optional[torch.device] = None):
        super().__init__(NX(nn.Parameter(torch.tensor(p, dtype=torch.float32, device=device), requires_grad=True)))
        self.weight = torch.tensor(lambda_value, dtype=torch.float32, device=device, requires_grad=False)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([factor], dim=1)
        return self.weight * torch.sum(norm_values)

    def project_(self):
        self.regularizer.p.data.clamp_(0.0)
        self.weight.data.clamp_(0.0)

    def values_(self) -> Dict[str, float]:
        return {'weight': self.weight.item(), 'p': self.regularizer.p.item()}


class ConstantAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer):
        super().__init__(regularizer)
        self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([factor], dim=1)
        return self.weight * torch.sum(norm_values)

    def project_(self):
        self.weight.data.clamp_(0.0)

    def values_(self) -> Dict[str, float]:
        return {'weight': self.weight.item()}


class EmbeddingAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 nb_objects: int):
        super().__init__(regularizer)
        self.weights = nn.Embedding(nb_objects, 1)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([factor], dim=1)
        weights = self.weights(features).view(-1)
        assert weights.shape == norm_values.shape
        return torch.sum(weights * norm_values)

    def project_(self):
        self.weights.weight.data.clamp_(0.0)

    def values_(self) -> Dict[str, float]:
        ni = self.weights.weight.shape[0]
        tmp = self.weights.weight.data.cpu().numpy()[:, 0].tolist()
        return {f'weight{i}': tmp[i] for i in range(ni)}


class LinearAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 nb_features: int):
        super().__init__(regularizer)
        self.projection = nn.Linear(nb_features, 1)
        with torch.no_grad():
            nn.init.zeros_(self.projection.weight)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        weight_values = torch.relu(self.projection(features)).view(-1)
        norm_values = self.regularizer([factor], dim=1)
        return torch.sum(weight_values * norm_values)

    def project_(self):
        pass

    def values_(self) -> Dict[str, float]:
        nf = self.projection.in_features
        tmp = self.projection.weight.data.cpu().numpy()[0, :].tolist()
        return {f'weight{i}': tmp[i] for i in range(nf)}


class GatedLinearAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 nb_features: int):
        super().__init__(regularizer)
        self.projection = nn.Linear(nb_features, 1)
        self.gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        weight_values = torch.relu(self.projection(features)).view(-1)
        norm_values = self.regularizer([factor], dim=1)
        res = torch.sum(weight_values * norm_values)
        return self.gate * res

    def project_(self):
        self.gate.data.clamp_(0)

    def values_(self) -> Dict[str, float]:
        nf = self.projection.in_features
        tmp = self.projection.weight.data.cpu().numpy()[0, :].tolist()
        res = {f'weight{i}': tmp[i] for i in range(nf)}
        res['gate'] = self.gate.item()
        return res


class GatedLinearSigmoidAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 nb_features: int):
        super().__init__(regularizer)
        self.projection = nn.Linear(nb_features, 1)
        self.gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        weight_values = torch.sigmoid(self.projection(features)).view(-1)
        norm_values = self.regularizer([factor], dim=1)
        res = torch.sum(weight_values * norm_values)
        return self.gate * res

    def project_(self):
        self.gate.data.clamp_(0)

    def values_(self) -> Dict[str, float]:
        nf = self.projection.in_features
        tmp = self.projection.weight.data.cpu().numpy()[0, :].tolist()
        res = {f'weight{i}': tmp[i] for i in range(nf)}
        res['gate'] = self.gate.item()
        return res
