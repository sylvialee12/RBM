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
        self.nspin = nspins
        self.hfield = hfield
        self.pbc = pbc

    def energy(self,config):
        """
        Finding the nonzero configuration for given configuration,
        which satisfies <s'|H|s> is nonzero
        """


        ess = -sum(config[1:]*config[0:-1]) - self.pbc*config[-1]*config[0]

        pass



