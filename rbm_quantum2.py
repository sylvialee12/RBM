__author__ = 'xlibb'

import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import Ising1D

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class rbm_qmc:
    """
    This is a variational monte carlo implementation on the Neural Network
    representing ground state.
    It is the Python implementation with Theano on the Science paper
    Science 355 ,602-606 (2017).
    For the details, please refer to the original paper.
    """
    def __init__(self,
                 nvisible,
                 nhidden,
                 hbias=None,
                 vbias=None,
                 W_real=None,
                 W_imag=None,
                 input=None,
                 np_rng=None,
                 theano_rng=None,
                 ):
        """
        RBM constructor,
        :param nvisible: number of visible nodes
        :param nhidden: number of hidden nodes
        :param hbias:"magnetic" field in the hidden layer,
                      if the value is None, then initialize it the random number generator
                      otherwise the initialized value is set to be the value of it.
        :param vbias: "magnetic" field in the visible layer,
                      if the value is None, then initialize it the random number generator
                      otherwise the initialized value is set to be the value of it.

        :param W_real: real part of the weight matrix connects visible layer and hidden layer

        :param W_imag: imaginary part of the weight matrix connects visible layer and hidden layer

        :param input: the initial sample for the visible layer (or spin configuration)
                      if the value is None, then initialize it with the random number generator
        :param np_rng: random number generator seed
        :param theano_rng: random number generator seed of Theano

        """
        pass


    def prop_up(self,vsample):
        pass
