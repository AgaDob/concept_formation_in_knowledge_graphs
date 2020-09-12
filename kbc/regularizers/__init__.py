# -*- coding: utf-8 -*-

from kbc.regularizers.base import Regularizer

from kbc.regularizers.base import F2
from kbc.regularizers.base import L1
from kbc.regularizers.base import N3
from kbc.regularizers.base import NX

from kbc.regularizers.adaptive import AdaptiveRegularizer

from kbc.regularizers.adaptive import NXAdaptiveRegularizer
from kbc.regularizers.adaptive import FixedLambdaNXAdaptiveRegularizer
from kbc.regularizers.adaptive import ConstantAdaptiveRegularizer
from kbc.regularizers.adaptive import EmbeddingAdaptiveRegularizer

from kbc.regularizers.adaptive import LinearAdaptiveRegularizer
from kbc.regularizers.adaptive import GatedLinearAdaptiveRegularizer
from kbc.regularizers.adaptive import GatedLinearSigmoidAdaptiveRegularizer

from kbc.regularizers.bregman import DiagonalMahalanobisAdaptiveRegularizer
from kbc.regularizers.bregman import ProjectedDiagonalMahalanobisAdaptiveRegularizer

__all__ = [
    'Regularizer',
    'F2',
    'L1',
    'N3',
    'NX',
    'AdaptiveRegularizer',
    'NXAdaptiveRegularizer',
    'FixedLambdaNXAdaptiveRegularizer',
    'ConstantAdaptiveRegularizer',
    'EmbeddingAdaptiveRegularizer',
    'LinearAdaptiveRegularizer',
    'GatedLinearAdaptiveRegularizer',
    'GatedLinearSigmoidAdaptiveRegularizer',
    'DiagonalMahalanobisAdaptiveRegularizer',
    'ProjectedDiagonalMahalanobisAdaptiveRegularizer'
]
