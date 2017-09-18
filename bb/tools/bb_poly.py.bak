"""
Bayesian Block implementation
=============================

Dynamic programming algorithm for finding the optimal adaptive-width histogram.

Based on Scargle et al 2012 [1]_

References
----------
.. [1] http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
"""
from __future__ import division
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from sympy.functions import re
from sympy.mpmath import *
from scipy import optimize
# TODO: implement other fitness functions from appendix B of Scargle 2012


class FitnessFunc(object):
    """Base class for fitness functions

    Each fitness function class has the following:
    - fitness(...) : compute fitness function.
       Arguments accepted by fitness must be among [T_k, N_k, a_k, b_k, c_k]
    - prior(N, Ntot) : compute prior on N given a total number of points Ntot
    """
    def __init__(self, p0=0.05, gamma=None):
        self.p0 = p0
        self.gamma = gamma

    def validate_input(self, t, x, sigma):
        """Check that input is valid"""
        pass

    def fitness(**kwargs):
        raise NotImplementedError()

    def prior(self, N, Ntot):
        if self.gamma is None:
            return self.p0_prior(N, Ntot)
        else:
            return self.gamma_prior(N, Ntot)

    def p0_prior(self, N, Ntot):
        # eq. 21 from Scargle 2012
        return 4 - np.log(73.53 * self.p0 * (N ** -0.478))

    def gamma_prior(self, N, Ntot):
        """Basic prior, parametrized by gamma (eq. 3 in Scargle 2012)"""
        if self.gamma == 1:
            return 0
        else:
            return (np.log(1 - self.gamma)
                    - np.log(1 - self.gamma ** (Ntot + 1))
                    + N * np.log(self.gamma))

    # the fitness_args property will return the list of arguments accepted by
    # the method fitness().  This allows more efficient computation below.
    @property
    def args(self):
        try:
            # Python 2
            return self.fitness.func_code.co_varnames[1:]
        except AttributeError:
            return self.fitness.__code__.co_varnames[1:]


class PolyEvents(FitnessFunc):
    """Fitness for binned or unbinned events that follow a polynomial
    piecewise distribution

    Parameters
    ----------
    p0 : float
        False alarm probability, used to compute the prior on N
        (see eq. 21 of Scargle 2012).  Default prior is for p0 = 0.
    gamma : float or None
        If specified, then use this gamma to compute the general prior form,
        p ~ gamma^N.  If gamma is specified, p0 is ignored.
    """
    def fitness(self, N_k, M_k, N_i):
        verbose = True
        # eq. 19 from Scargle 2012
        #a = (N_k-2)/(T_k*(N_k-1))
        if(verbose):
          print 'fitness call'
          print 'N_i',N_i
          print 'N_k',N_k
          print 'M_k',M_k
        #raw_input()
        x = Symbol('x')
        a_i = []

        def f_a(a,M_k,N_k,N_i,i):
          return 2/M_k[i] - a + (N_k[i]) * np.sum((N_i[i:-1]-N_i[-1])/(1+a*(N_i[i:-1]-N_i[-1])))**-1
          #return 2/M_k[i] - a + (N_k[i]) * np.sum((N_i[i+1:]-N_i[-1])/(1+a*(N_i[i+1:]-N_i[-1])))**-1

        for i in range(len(N_i)):

        #special cases
          if N_k[i]==1:
            a_i.append(np.inf)
          elif N_k[i]==2:
            a_i.append(0)

          else:

            #upper_bound = 2.0/M_k[i]
            upper_bound= min(1.0/(N_i[-1]-N_i[0]),2.0/M_k[i])
            #upper_bound= max(1.0/(N_i[-1]-N_i[-2]),2.0/M_k[i])*2
            lower_bound = upper_bound
            #start_val = (lower_bound+upper_bound)/2.0
            start_val = upper_bound/10.0


            while(np.sign(f_a(upper_bound,M_k,N_k,N_i,i))==np.sign(f_a(lower_bound,M_k,N_k,N_i,i))):
              lower_bound-=0.1
            if verbose:
              print 'a_i loop'
              print 'M_k', M_k[i]
              print 'N_k', N_k[i]
              print 'N_i', N_i
              print 'upper', upper_bound, f_a(upper_bound,M_k,N_k,N_i,i)
              print 'lower', lower_bound, f_a(lower_bound,M_k,N_k,N_i,i)
              print f_a(x,M_k,N_k,N_i,i)

            #a_sol = optimize.newton(f_a,start_val,args=(M_k,N_k,N_i,i),maxiter=5000)
            a_sol = optimize.brentq(f_a,lower_bound,upper_bound,args=(M_k,N_k,N_i,i),maxiter=5000)
            #a_sol = start_val
            #print a_sol
            #a_sol = a_sol.x[0]


            #f2 = lambda a: f_a(a,M_k,N_k,N_i,i)
            #a_sol = findroot(lambda a: 2/M_k[i] - a + N_k[i] * np.sum((N_i[i:-1]-N_i[-1])/(1+a*(N_i[i:-1]-N_i[-1])))**-1,start_val)
            #a_sol = findroot(f2,start_val+1,tol=0.0001,solver = 'newton')
            #a_sol = findroot(f2,start_val, tol = 0.0001,solver='halley')
            #a_sol = findroot(f2,start_val,solver='anderson')
            #start_val = a_sol
            #a_sol = 1


            if verbose:
              print 'a_sol', a_sol
              raw_input()

            a_i.append(a_sol)

        #if len(N_i)>1:
        a_i = np.asarray(a_i,dtype=float)
        lamb = (N_k)/((M_k)*(1-a_i*(M_k)/2.0))
        lamb = np.where(np.isinf(a_i),1,lamb) #if a is inf, lambda is 0
        if verbose:
          print 'calculating lambda'
          print 'a_i',a_i
          print 'N_k',N_k
          print 'M_k',M_k
          print 'lamb', lamb
        loglamb = np.log(lamb)
        if np.any(np.isnan(loglamb)):
          print 'loglamb nan, man:',loglamb
        #loglamb = np.where(np.isnan(loglamb),-100,loglamb)
        #print 'lamb',lamb, loglamb
        #raw_input()
        logsum = []
        for i in range(len(N_i)):
          logsum.append(np.sum(np.log(1+(a_i[i])*(N_i[i:-1]-N_i[-1])))+np.log(1))
        logsum = np.asarray(logsum)
        if verbose:
          print 'calculating logsum'
          print 'a_i',a_i
          print 'N_i',N_i
          print 'logsum',logsum
          print 'slope (lambda*a)', lamb*a_i
          print 'y_in', lamb*(1-a_i*M_k)
          print 'y_fin', lamb
        if np.any(np.isnan(logsum)):
          print a_i, N_i[:-1],N_i[-1]
          print 'logsum nan:',1+(a_i[i])*(N_i[i:-1]-N_i[-1])
        #return N_k * loglamb + N_k * np.where(np.isnan(logsum),-100,logsum) - N_k
        y_in = lamb*(1-a_i*M_k)
        y_fin = lamb
        return ((N_k) * loglamb + (N_k) * logsum, y_in, y_fin)
        #return (N_k+1) * logsum
        #return (lamb**N_k+np.prod(1+a_i*(N_i-N_i[-1])))/10.0**100

        #return N_k * loglamb + N_k * np.where(np.isnan(np.log(1+a_i*(M_k))),-100,np.log(1+a_i*(M_k)))
        #return N_k * loglamb

    def prior(self, N, Ntot):
        if self.gamma is not None:
            return self.gamma_prior(N, Ntot)
        else:
            # eq. 21 from Scargle 2012
            return 4 - np.log(73.53 * self.p0 * (N ** -0.478))

def bayesian_blocks(t, x=None, sigma=None,
                    fitness='poly_events', gamma=None, p0=0.05):
    """Bayesian Blocks Implementation

    This is a flexible implementation of the Bayesian Blocks algorithm
    described in Scargle 2012 [1]_

    Parameters
    ----------
    t : array_like
        data times (one dimensional, length N)
    x : array_like (optional)
        data values
    sigma : array_like or float (optional)
        data errors
    fitness : str or object
        the fitness function to use.
        If a string, the following options are supported:

        - 'events' : binned or unbinned event data
            extra arguments are `p0`, which gives the false alarm probability
            to compute the prior, or `gamma` which gives the slope of the
            prior on the number of bins.
        - 'regular_events' : non-overlapping events measured at multiples
            of a fundamental tick rate, `dt`, which must be specified as an
            additional argument.  The prior can be specified through `gamma`,
            which gives the slope of the prior on the number of bins.
        - 'measures' : fitness for a measured sequence with Gaussian errors
            The prior can be specified using `gamma`, which gives the slope
            of the prior on the number of bins.  If `gamma` is not specified,
            then a simulation-derived prior will be used.

        Alternatively, the fitness can be a user-specified object of
        type derived from the FitnessFunc class.

    Returns
    -------
    edges : ndarray
        array containing the (N+1) bin edges

    Examples
    --------
    Event data:

    >>> t = np.random.normal(size=100)
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Event data with repeats:

    >>> t = np.random.normal(size=100)
    >>> t[80:] = t[:20]
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Regular event data:

    >>> dt = 0.01
    >>> t = dt * np.arange(1000)
    >>> x = np.zeros(len(t))
    >>> x[np.random.randint(0, len(t), len(t) / 10)] = 1
    >>> bins = bayesian_blocks(t, fitness='regular_events', dt=dt, gamma=0.9)

    Measured point data with errors:

    >>> t = 100 * np.random.random(100)
    >>> x = np.exp(-0.5 * (t - 50) ** 2)
    >>> sigma = 0.1
    >>> x_obs = np.random.normal(x, sigma)
    >>> bins = bayesian_blocks(t, fitness='measures')

    References
    ----------
    .. [1] Scargle, J `et al.` (2012)
           http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    See Also
    --------
    astroML.plotting.hist : histogram plotting function which can make use
                            of bayesian blocks.
    """
    # validate array input
    t = np.asarray(t, dtype=float)
    if x is not None:
        x = np.asarray(x)
    if sigma is not None:
        sigma = np.asarray(sigma)

    # verify the fitness function
    if fitness == 'poly_events':
        if x is not None and np.any(x % 1 > 0):
            raise ValueError("x must be integer counts for fitness='events'")
        fitfunc = PolyEvents(p0,gamma)
    else:
        if not (hasattr(fitness, 'args') and
                hasattr(fitness, 'fitness') and
                hasattr(fitness, 'prior')):
            raise ValueError("fitness not understood")
        fitfunc = fitness

    # find unique values of t
    t = np.array(t, dtype=float)
    assert t.ndim == 1
    unq_t, unq_ind, unq_inv = np.unique(t, return_index=True,
                                        return_inverse=True)

    # if x is not specified, x will be counts at each time
    if x is None:
        if sigma is not None:
            raise ValueError("If sigma is specified, x must be specified")

        if len(unq_t) == len(t):
            x = np.ones_like(t)
        else:
            x = np.bincount(unq_inv)

        t = unq_t
        sigma = 1

    # if x is specified, then we need to sort t and x together
    else:
        x = np.asarray(x)

        if len(t) != len(x):
            raise ValueError("Size of t and x does not match")

        if len(unq_t) != len(t):
            raise ValueError("Repeated values in t not supported when "
                             "x is specified")
        t = unq_t
        x = x[unq_ind]

    # verify the given sigma value
    N = t.size
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.shape not in [(), (1,), (N,)]:
            raise ValueError('sigma does not match the shape of x')
    else:
        sigma = 1

    # validate the input
    fitfunc.validate_input(t, x, sigma)

    # compute values needed for computation, below
    if 'a_k' in fitfunc.args:
        ak_raw = np.ones_like(x) / sigma / sigma
    if 'b_k' in fitfunc.args:
        bk_raw = x / sigma / sigma
    if 'c_k' in fitfunc.args:
        ck_raw = x * x / sigma / sigma

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    y_fin_prev = 0
    cont_diff = 0
    for R in range(N):
        print R, t[R]
        # Compute fit_vec : fitness of putative last block (end at R)
        kwds = {}

        # T_k: width/duration of each block
        if 'T_k' in fitfunc.args:
            kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]

        # M_k: width of each block, assuming points are located at bin edges
        if 'M_k' in fitfunc.args:
          kwds['M_k'] = t[R]-t[:R+1]
          #kwds['M_k'] = block_length[:R + 1] - block_length[R + 1]

        # N_k: number of elements in each block
        if 'N_k' in fitfunc.args:
            kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]

        # a_k: eq. 31
        if 'a_k' in fitfunc.args:
            kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]

        # b_k: eq. 32
        if 'b_k' in fitfunc.args:
            kwds['b_k'] = - np.cumsum(bk_raw[:R + 1][::-1])[::-1]

        # c_k: eq. 33
        if 'c_k' in fitfunc.args:
            kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]

        # N_i: all block elements
        if 'N_i' in fitfunc.args:
          kwds['N_i'] = t[:R+1]
          #kwds['N_i'] = block_length[:R+1]

        # evaluate fitness function
        (fit_vec,y_in,y_fin) = fitfunc.fitness(**kwds)
        if R>1:
          cont_diff = np.abs(y_fin_prev - y_in)
          cont_diff[np.isnan(cont_diff)]=1e20

        y_fin_prev = np.concatenate([y_fin,[0]])
        print 'y_in', y_in
        print 'y_fin', y_fin
        print 'cont_diff',cont_diff
        print 'cont_diff exp',np.exp(cont_diff)

        A_R = fit_vec - fitfunc.prior(R + 1, N) - np.exp(cont_diff)**2
        #A_R = fit_vec
        print 'A_R', A_R
        A_R[1:] += best[:R]
        print 'A_R after best', A_R

        i_max = np.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]
        print 'last:',last
        print 'best:',best
        #raw_input()

    #-----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]
    change_points[-1] = change_points[-1]-1 # temp line for using t instead of edges
    print 'change_points',change_points

    #return edges[change_points]
    return t[change_points]
