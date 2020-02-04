from abc import ABCMeta, abstractmethod

import numpy as np


class ActivationFunction(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def function(self, x):
        return

    @abstractmethod
    def prime(self, x):
        return


class SigmoidActivationFunction(ActivationFunction):
    @classmethod
    def function(cls, x):

        return 1 / (1.0 + np.exp(-x))

    @classmethod
    def prime(cls, x):

        return x * (1 - x)


class ReLUActivationFunction(ActivationFunction):
    @classmethod
    def function(cls, x):

        return np.maximum(np.zeros(x.shape), x)

    @classmethod
    def prime(cls, x):

        return (x > 0).astype(int)


class TanhActivationFunction(ActivationFunction):
    @classmethod
    def function(cls, x):

        return np.tanh(x)

    @classmethod
    def prime(cls, x):

        return 1 - x * x
