def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width_uncor = t[K]-t[:K+1] #same as M_k
        width_cor = block_length[:K + 1] - block_length[K + 1] #same as T_k
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1] #same as N_k
        elems = t[:R+1] #same as N_i

        # evaluate fitness function for these possibilities
        fit_vec = fitness(count_vec, width_uncor, elems)
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]

def fitness(N_k, M_k, N_i):
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
    y_in = lamb*(1-a_i*M_k)
    y_fin = lamb
    return ((N_k) * loglamb + (N_k) * logsum, y_in, y_fin)
