import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# This works normally.

# ---------------------------------------------------------
# rBergomi model class
# ---------------------------------------------------------

def dW1(e, c, N, s):
    """
    Produces random numbers for variance process with required
    covariance structure.
    """
    rng = np.random.multivariate_normal
    return rng(e, c, (N, s))

def Y(N, s, n, a, dW):
    """
    Constructs Volterra process from appropriately
    correlated 2d Brownian increments.
    """
    Y1 = np.zeros((N, 1 + s)) # Exact integrals
    Y2 = np.zeros((N, 1 + s)) # Riemann sums

    # Construct Y1 through exact integral
    for i in np.arange(1, 1 + s, 1):
        Y1[:,i] = dW[:,i-1,1] # Assumes kappa = 1

    # Construct arrays for convolution
    G = np.zeros(1 + s) # Gamma
    for k in np.arange(2, 1 + s, 1):
        G[k] = g(b(k, a)/n, a)

    X = dW[:,:,0] # Xi

    # Initialise convolution result, GX
    GX = np.zeros((N, len(X[0,:]) + len(G) - 1))

    # Compute convolution, FFT not used for small n
    # Possible to compute for all paths in C-layer?
    for i in range(N):
        GX[i,:] = np.convolve(G, X[i,:])

    # Extract appropriate part of convolution
    Y2 = GX[:,:1 + s]

    # Finally contruct and return full process
    Y = np.sqrt(2 * a + 1) * (Y1 + Y2)
    return Y

def dW2(N, s, dt):
    """
    Obtain orthogonal increments.
    """
    return np.random.randn(N, s) * np.sqrt(dt)

def dB(dW1, dW2, rho = 0.0):
    """
    Constructs correlated price Brownian increments, dB.
    """
    rho = rho
    dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
    return dB

def V(Y, xi, eta, t, a):
    """
    rBergomi variance process.
    """
    V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
    return V

def S(V, dB, S0, dt):
    """
    rBergomi price process.
    """
    # Construct non-anticipative Riemann increments
    increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = np.cumsum(increments, axis = 1)

    S = np.zeros_like(V)
    S[:,0] = S0
    S[:,1:] = S0 * np.exp(integral)
    return S

def S1(V, dW1, rho, S0, dt):
    """
    rBergomi parallel price process.
    """

    # Construct non-anticipative Riemann increments
    increments = rho * np.sqrt(V[:,:-1]) * dW1[:,:,0] - 0.5 * rho**2 * V[:,:-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = np.cumsum(increments, axis = 1)

    S = np.zeros_like(V)
    S[:,0] = S0
    S[:,1:] = S0 * np.exp(integral)
    return S

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s