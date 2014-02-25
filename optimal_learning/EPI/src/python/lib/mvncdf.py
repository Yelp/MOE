'''multivariate normal probabilities and cumulative distribution function
a wrapper for scipy.stats.kde.mvndst


      SUBROUTINE MVNDST( N, LOWER, UPPER, INFIN, CORREL, MAXPTS,
     &                   ABSEPS, RELEPS, ERROR, VALUE, INFORM )
*
*     A subroutine for computing multivariate normal probabilities.
*     This subroutine uses an algorithm given in the paper
*     "Numerical Computation of Multivariate Normal Probabilities", in
*     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
*          Alan Genz
*          Department of Mathematics
*          Washington State University
*          Pullman, WA 99164-3113
*          Email : AlanGenz@wsu.edu
*
*  Parameters
*
*     N      INTEGER, the number of variables.
*     LOWER  REAL, array of lower integration limits.
*     UPPER  REAL, array of upper integration limits.
*     INFIN  INTEGER, array of integration limits flags:
*            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
*     CORREL REAL, array of correlation coefficients; the correlation
*            coefficient in row I column J of the correlation matrix
*            should be stored in CORREL( J + ((I-2)*(I-1))/2 ), for J < I.
*            THe correlation matrix must be positive semidefinite.
*     MAXPTS INTEGER, maximum number of function values allowed. This
*            parameter can be used to limit the time. A sensible
*            strategy is to start with MAXPTS = 1000*N, and then
*            increase MAXPTS if ERROR is too large.
*     ABSEPS REAL absolute error tolerance.
*     RELEPS REAL relative error tolerance.
*     ERROR  REAL estimated absolute error, with 99% confidence level.
*     VALUE  REAL estimated value for the integral
*     INFORM INTEGER, termination status parameter:
*            if INFORM = 0, normal completion with ERROR < EPS;
*            if INFORM = 1, completion with ERROR > EPS and MAXPTS
*                           function vaules used; increase MAXPTS to
*                           decrease ERROR;
*            if INFORM = 2, N > 500 or N < 1.
*



>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[10.0,10.0],[0,0],[0.5])
(2e-016, 1.0, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[100.0,100.0],[0,0],[0.0])
(2e-016, 1.0, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[1.0,1.0],[0,0],[0.0])
(2e-016, 0.70786098173714096, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.001,1.0],[0,0],[0.0])
(2e-016, 0.42100802096993045, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.001,10.0],[0,0],[0.0])
(2e-016, 0.50039894221391101, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.001,100.0],[0,0],[0.0])
(2e-016, 0.50039894221391101, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.01,100.0],[0,0],[0.0])
(2e-016, 0.5039893563146316, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.1,100.0],[0,0],[0.0])
(2e-016, 0.53982783727702899, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.1,100.0],[2,2],[0.0])
(2e-016, 0.019913918638514494, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.0,0.0],[0,0],[0.0])
(2e-016, 0.25, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.0,0.0],[-1,0],[0.0])
(2e-016, 0.5, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.0,0.0],[-1,0],[0.5])
(2e-016, 0.5, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.0,0.0],[0,0],[0.5])
(2e-016, 0.33333333333333337, 0)
>>> scipy.stats.kde.mvn.mvndst([0.0,0.0],[0.0,0.0],[0,0],[0.99])
(2e-016, 0.47747329317779391, 0)
'''
import numpy as np
import scipy
import scipy.stats

informcode = {0: 'normal completion with ERROR < EPS',
              1: '''completion with ERROR > EPS and MAXPTS function values used;
                    increase MAXPTS to decrease ERROR;''',
              2: 'N > 500 or N < 1'}

def mvstdnormcdf(lower, upper, corrcoef, **kwds):
    '''standardized multivariate normal cumulative distribution function

    This is a wrapper for scipy.stats.kde.mvn.mvndst which calculates
    a rectangular integral over a standardized multivariate normal
    distribution.

    This function assumes standardized scale, that is the variance in each dimension
    is one, but correlation can be arbitrary, covariance = correlation matrix

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    corrcoef : float or array_like
       specifies correlation matrix in one of three ways, see notes
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.

    Returns
    -------
    cdfvalue : float
        value of the integral


    Notes
    -----
    The correlation matrix corrcoef can be given in 3 different ways
    If the multivariate normal is two-dimensional than only the
    correlation coefficient needs to be provided.
    For general dimension the correlation matrix can be provided either
    as a one-dimensional array of the upper triangular correlation
    coefficients stacked by rows, or as full square correlation matrix

    See Also
    --------
    mvnormcdf : cdf of multivariate normal distribution without
        standardization

    Examples
    --------

    >>> print mvstdnormcdf([-np.inf,-np.inf], [0.0,np.inf], 0.5)
    0.5
    >>> corr = [[1.0, 0, 0.5],[0,1,0],[0.5,0,1]]
    >>> print mvstdnormcdf([-np.inf,-np.inf,-100.0], [0.0,0.0,0.0], corr, abseps=1e-6)
    0.166666399198
    >>> print mvstdnormcdf([-np.inf,-np.inf,-100.0],[0.0,0.0,0.0],corr, abseps=1e-8)
    something wrong completion with ERROR > EPS and MAXPTS function values used;
                        increase MAXPTS to decrease ERROR; 1.048330348e-006
    0.166666546218
    >>> print mvstdnormcdf([-np.inf,-np.inf,-100.0],[0.0,0.0,0.0], corr,
                            maxpts=100000, abseps=1e-8)
    0.166666588293

    '''
    n = len(lower)
    #don't know if converting to array is necessary,
    #but it makes ndim check possible
    lower = np.array(lower)
    upper = np.array(upper)
    corrcoef = np.array(corrcoef)

    correl = np.zeros(n*(n-1)/2.0)  #dtype necessary?

    if (lower.ndim != 1) or (upper.ndim != 1):
        raise ValueError, 'can handle only 1D bounds'
    if len(upper) != n:
        raise ValueError, 'bounds have different lengths'
    if n==2 and corrcoef.size==1:
        correl = corrcoef
        #print 'case scalar rho', n
    elif corrcoef.ndim == 1 and len(corrcoef) == n*(n-1)/2.0:
        #print 'case flat corr', corrcoeff.shape
        correl = corrcoef
    elif corrcoef.shape == (n,n):
        #print 'case square corr',  correl.shape
        for ii in range(n):
            for jj in range(ii):
                correl[ jj + ((ii-2)*(ii-1))/2] = corrcoef[ii,jj]
    else:
        raise ValueError, 'corrcoef has incorrect dimension'

    if not 'maxpts' in kwds:
        if n >2:
            kwds['maxpts'] = 10000*n

    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)
    infin = 2.0*np.ones(n)

    np.putmask(infin,lowinf,0)# infin.putmask(0,lowinf)
    np.putmask(infin,uppinf,1) #infin.putmask(1,uppinf)
    #this has to be last
    np.putmask(infin,lowinf*uppinf,-1)

##    #remove infs
##    np.putmask(lower,lowinf,-100)# infin.putmask(0,lowinf)
##    np.putmask(upper,uppinf,100) #infin.putmask(1,uppinf)

    #print lower,',',upper,',',infin,',',correl
    #print correl.shape
    #print kwds.items()
    error, cdfvalue, inform = scipy.stats.kde.mvn.mvndst(lower,upper,infin,correl,**kwds)
    if inform:
        print 'something wrong', informcode[inform], error
    return cdfvalue


def mvnormcdf(lower, upper, mu, cov, **kwds):
    '''multivariate normal cumulative distribution function

    This is a wrapper for scipy.stats.kde.mvn.mvndst which calculates
    a rectangular integral over a multivariate normal distribution.

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    mu : array_lik, 1d
       list or array of means
    cov : array_like, 2d
       specifies covariance matrix
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.

    Returns
    -------
    cdfvalue : float
        value of the integral


    Notes
    -----
    This function normalizes the location and scale of the multivariate
    normal distribution and then uses `mvstdnormcdf` to call the integration.

    See Also
    --------
    mvstdnormcdf : location and scale standardized multivariate normal cdf
    '''

    lower = np.array(lower)
    upper = np.array(upper)
    cov = np.array(cov)
    stdev = np.sqrt(np.diag(cov)) # standard deviation vector
    #do I need to make sure stdev is float and not int?
    #is this correct to normalize to corr?
    lower = (lower - mu)/stdev
    upper = (upper - mu)/stdev
    divrow = np.atleast_2d(stdev)
    corr = cov/divrow/divrow.T
    #v/np.sqrt(np.atleast_2d(np.diag(covv)))/np.sqrt(np.atleast_2d(np.diag(covv))).T

    return mvstdnormcdf(lower, upper, corr, **kwds)
