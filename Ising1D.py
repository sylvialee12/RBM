__author__ = 'xlibb'
import numpy as np
import theano.tensor as T

class Ising1D:
    """
    Ising 1D system, giving the systems size, tranverse field strength
    """
    def __init__(self,nspins,hfield,pbc=1):
        """

        """
        self.nspin=nspins
        self.hfield=hfield
        self.pbc=pbc

    def findcon(self,config):
        """
        Finding the nonzero configuration for given configuration,
        which satisfies <s'|H|s> is nonzero
        """
        pass

