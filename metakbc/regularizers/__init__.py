# -*- coding: utf-8 -*-

from metakbc.regularizers.base import Regularizer

from metakbc.regularizers.base import F2
from metakbc.regularizers.base import L1
from metakbc.regularizers.base import N3
from metakbc.regularizers.base import NX

from metakbc.regularizers.adaptive import AdaptiveRegularizer

from metakbc.regularizers.adaptive import NXAdaptiveRegularizer
from metakbc.regularizers.adaptive import FixedLambdaNXAdaptiveRegularizer
from metakbc.regularizers.adaptive import ConstantAdaptiveRegularizer
from metakbc.regularizers.adaptive import EmbeddingAdaptiveRegularizer

from metakbc.regularizers.adaptive import LinearAdaptiveRegularizer
from metakbc.regularizers.adaptive import GatedLinearAdaptiveRegularizer
from metakbc.regularizers.adaptive import GatedLinearSigmoidAdaptiveRegularizer

from metakbc.regularizers.bregman import DiagonalMahalanobisAdaptiveRegularizer
from metakbc.regularizers.bregman import ProjectedDiagonalMahalanobisAdaptiveRegularizer

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
