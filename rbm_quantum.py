__author__ = 'xlibb'

import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import Ising1D

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class rbm_classicalsolver:
    """
    This is a VMC method based on Restricted Boltzmann Machine to find quantum many body ground state.
     It is the Python implementation with Theano on the Science paper Science 355 ,602¨C606 (2017).
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
        self.nvisible=nvisible
        self.nhidden=nhidden
        if np_rng is None:
            # create a number generator
            np_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if W_real is None:
            # W_real is initialized with `initial_Wreal` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_Wreal = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (nhidden + nvisible)),
                    high=4 * np.sqrt(6. / (nhidden + nvisible)),
                    size=(nvisible, nhidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights real part
            W_real = theano.shared(value=initial_Wreal, name='Wreal', borrow=True)
        if W_imag is None:
            # W_real is initialized with `initial_Wreal` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_Wimag = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (nhidden + nvisible)),
                    high=4 * np.sqrt(6. / (nhidden + nvisible)),
                    size=(nvisible, nhidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights imaginary part
            W_imag = theano.shared(value=initial_Wimag, name='Wimag', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=np.zeros(
                    nhidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=np.zeros(
                    nvisible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        # self.input = input
        # if not input:
        #     self.input = T.matrix('input')
        # self.input=input

        self.W_real = W_real
        self.W_imag=W_imag
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.input=input
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        #self.params = [self.W_real,self.W_imag,self.hbias, self.vbias]
        self.params=[self.W_real,self.hbias,self.vbias]


    def prop_up(self,vis):
        """
        Probability of Markov chain step for up propagation (from visible
        samples to hidden samples)

        :param vis: visible samples
        :return pre_activation: for later optimization
                activation_simoid value for transition probability P(h_i|vis)
        """
        pre_activation=(2.0*self.hbias)+T.dot(vis,2.0*self.W_real)
        return [pre_activation,T.nnet.sigmoid(pre_activation)]

    def sample_h_givenv(self,v0_sample):
        """
        Markov chain transition from given visible layer configuration to 
        generate hidden layer configuration
        :param v0_sample: given visible layer configurations
        :return h1_sample: hidden layer configurations
        """
        
        pre_sigmoid_h1,h1_mean = self.prop_up(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample=2*self.theano_rng.binomial(size=h1_mean.shape,
                                      n=1,
                                      p=h1_mean,
                                      dtype=theano.config.floatX
                                      )-1

        return [pre_sigmoid_h1,h1_mean,h1_sample]

    def prop_dn(self,hid):
        """
        Probability of Markov chain step for up propagation 
        (from visible samples to hidden samples)

        :param hid: hidden samples
        :return pre_activation: for later optimization
                activation_simoid value for transition probability P(vis_i|h)
        """

        pre_activation=(2.0*self.vbias)+T.dot(hid, 2.0*self.W_real.T)
        # or maybe it can be T.dot(hid, self.W.T)? wait for check
        return [pre_activation,T.nnet.sigmoid(pre_activation)]

    def sample_v_givenh(self,h0_sample):
        """
        Markov chain transition from given hidden layer configuration
        to generate visible layer configuration
        :param h0_sample: given visible layer configurations
        :return v1_sample: hidden layer configurations
        """
        pre_sigmoid_v1,v1_mean = self.prop_dn(h0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample=2*self.theano_rng.binomial(size=v1_mean.shape,
                                      n=1,
                                      p=v1_mean,
                                      dtype=theano.config.floatX
                                      )-1
        return [pre_sigmoid_v1,v1_mean,v1_sample]


    def gibbs_vhv(self,v0_sample):
        """
        Markov chain transition of visible sample->hidden sample->visible 
        sample with Gibbs sampling
        :param v0_sample: given visible layer configuration
        :return v1_sample: visible layer configuration after two 
        steps of Markov chain transition
        """
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_givenv(v0_sample)

        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_givenh(h1_sample)

        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]


    def gibbs_hvh(self,h0_sample):
        """
        Markov chain transition of hidden sample->visible sample->hidden 
        sample, with Gibbs sampling
        :param h0_sample: given hidden layer configuration
        :return h1_sample: hidden layer configuration after two steps of
        Markov chain transition
        """
        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_givenh(h0_sample)

        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_givenv(v1_sample)

        return [pre_sigmoid_v1,v1_mean,v1_sample,
                pre_sigmoid_h1,h1_mean,h1_sample]

    def changing_weight(self,v_sample):
        """
        function to compute the transition probability flipping spins for v_sample,
        which is Ts's=conj(psi(s',M))/conj(psi(s,M)),
        as for transverse field Ising model the flipping term in the Hamiltonian
        is hf=h/2(sp_i+sm_i), therefore flip each site contribute the same energy h/2,
        but the Ts's is different, but one can sum up Ts's for all s'
         :param v_sample: one sample of the visible layer
         :param Hamiltonian: Hamiltonian of the physical system we concern, 
         we mainly use Hamiltonian.h
         :pbc: periodic boundary condition, 1:periodic, 0: open

        """
        # As self.W_real has the size of nvisible*nhidden, and here v_sample is a
        # vector of nvisible, so ones needs to transpose self.W_real to make it broadcastable
        exponent=-2*v_sample*self.vbias+\
                 T.sum(
                     T.log(T.cosh(self.hbias-self.W_real*v_sample)),axis=1)-\
                 T.sum(
                     T.log(T.cosh(self.hbias+self.W_real*v_sample)),axis=1)
        return T.sum(T.exp(exponent))
        
    def changing_weight2(self,v_sample):
        """
        function to compute the transition probability flipping spins for v_sample,
        which is Ts's=conj(psi(s',M))/conj(psi(s,M)),
        as for transverse field Ising model the flipping term in the Hamiltonian
        is hf=h/2(sp_i+sm_i), therefore flip each site contribute the same energy h/2,
        but the Ts's is different, but one can sum up Ts's for all s'
         :param v_sample: one sample of the visible layer
         :param Hamiltonian: Hamiltonian of the physical system we concern, 
         we mainly use Hamiltonian.h
         :pbc: periodic boundary condition, 1:periodic, 0: open

        """
        # As self.W_real has the size of nvisible*nhidden, and here v_sample is a
        # vector of nvisible, so ones needs to transpose self.W_real to make it broadcastable
        exponent=-2*v_sample*self.vbias+\
                 T.sum(
                     T.log(T.cosh(self.hbias-T.shape_padaxis(self.W_real,axis=0)*T.shape_padaxis(v_sample,axis=-1))),axis=2)-\
                 T.sum(
                     T.log(T.cosh(self.hbias+T.shape_padaxis(self.W_real,axis=0)*T.shape_padaxis(v_sample,axis=-1))),axis=2)
                
        return T.mean(T.sum(T.exp(exponent),axis=1))




    def variational_energy(self,v1_sample,Hamiltonian):
        """
        function to compute flpping energy for one sample to the given sample, 
        which is sum_s'<s|H|s'>
         :param v_sample: sample of the visible layer
         :param Hamiltonian: Hamiltonian of the physical system we concern
        """
        # For transverse field Ising model, H= -s_i^s_j+h/2(sp_i+sm_i)
        #periodic boundary condition
        left_shift_sample,_=theano.map(lambda x:T.concatenate([x[1:],T.stack(Hamiltonian.pbc*x[0])]),
                                   sequences=[v1_sample])


        ess=T.sum(T.mean(-left_shift_sample*v1_sample,axis=0))
        #changingweight,_=theano.map(lambda x: self.changing_weight(x),sequences=[v1_sample])
        
        #esp=1/2*Hamiltonian.hfield*T.mean(changingweight)
        esp=1/2*Hamiltonian.hfield*self.changing_weight2(v1_sample)
        return ess+esp


    def variational_update(self,Hamiltonian,lr=0.1,persistent=None,k=1):
        """
        This is to update the Neural Network parameters with stochastic gradient descent method.
        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_givenv(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        variational=self.variational_energy(chain_end,Hamiltonian)
        sz=self.magnetization(chain_end)
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(variational, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        #if persistent:
            #updates[persistent]=nh_samples[-1]

        return updates,variational,sz

    def magnetization(self,v1_sample):
        """
        Get the magnetization of the variational wave function through sampling,
        the samples must be the end of the Markov chain
        """
        sz=T.mean(v1_sample)
        return sz


def testRBM_classical():
    """
    """
    hfield=0.1
    nspin=100
    alpha=2
    n_sample=1000
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    ising1d=Ising1D.Ising1D(nspin,hfield)
    initial_sample=theano.shared(2*rng.binomial(size=(n_sample,nspin),
                                          n=1,p=0.5).astype('float64')-1,borrow=True)
    rbm_classical=rbm_classicalsolver(nvisible=nspin,nhidden=alpha*nspin,
                                      input=initial_sample,
                                      np_rng=rng,theano_rng=theano_rng)

    updates,variational_energy,sz=rbm_classical.variational_update(ising1d,
                                                                persistent=None,
                                                                k=1)
    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    #################################
    #     Training the RBM          #
    #################################
    index=T.iscalar('index')                                                            
    train_rbm = theano.function(
        [],
        outputs=[variational_energy,sz],
        updates=updates,
        name='train_rbm'
    )

    times=200
    for epoch in range(times):
        # go through the training set
        print(epoch)
        [mean_cost,sz] = train_rbm()
        print("For epoch %d , the variational energy is %f, /n magnetization is %f"
        %(epoch,mean_cost,sz))


if __name__=="__main__":
    testRBM_classical()
