import jax.numpy as jnp
from jax import random, lax
from jax.scipy.stats import norm
import jaxopt as jo
from jax import jit

# ---------------------------------------------------------
# rBergomi model class
# ---------------------------------------------------------

def dW1(key, e, c, N, s):
    """
    Produces random numbers for variance process with required
    covariance structure.
    """
    key_, subkey = random.split(key)
    dW1 = random.multivariate_normal(subkey, e, c, (N, s))
    return dW1, key_

def Y(N, s, n, a, dW):
    """
    Constructs Volterra process from appropriately correlated 2D Brownian increments. Uses FFT for convolution.
    Parameters:
    - N: number of paths
    - s: steps in time
    - n: convolution kernel length (typically 365 in your case)
    - a: roughness parameter
    - dW: Brownian increments
    """
    Y1 = jnp.zeros((N, 1 + s))  # Exact integrals
    def y1(i, Y1):  # Construct Y1 through exact integral
        Y1 = Y1.at[:, i].set(dW[:, i - 1, 1])  # Assuming kappa = 1
        return Y1
    
    Y1 = lax.fori_loop(1, 1 + s, y1, Y1)
    
    Y2 = jnp.zeros((N, 1 + s))  # Riemann sums

    # Construct arrays for convolution
    G = jnp.zeros(1 + s)  # Gamma array
    def pre_generate_array(k, G):
        G = G.at[k].set(g(b(k, a) / n, a))
        return G

    G = lax.fori_loop(2, 1 + s, pre_generate_array, G)
    X = dW[:, :, 0]  # Xi from the Brownian increments

    # Compute convolution: option to use FFT-based or direct convolution
    GX = jnp.zeros((N, len(X[0, :]) + len(G) - 1))
    
    def convolve_row_fft(i, GX):
        GX = GX.at[i, :].set(fft_convolve(G, X[i, :]))
        return GX
    GX = lax.fori_loop(0, N, convolve_row_fft, GX)

    # Extract appropriate part of convolution
    Y2 = GX[:, :1 + s]

    # Finally, construct and return the full process
    Y = jnp.sqrt(2 * a + 1) * (Y1 + Y2)
    return Y

def dW2(key, N, s, dt):
    """
    Obtain orthogonal increments.
    """
    key_, subkey = random.split(key)
    dW2 = random.normal(subkey, (N, s)) * jnp.sqrt(dt)
    return dW2, key_

def dB(dW1, dW2, rho):
    """
    Constructs correlated price Brownian increments, dB.
    """
    return rho * dW1[:, :, 0] + jnp.sqrt(1 - rho ** 2) * dW2

def V(Y, xi, eta, t, a):
    """
    rBergomi variance process.
    """
    return xi * jnp.exp(eta * Y - 0.5 * eta ** 2 * t ** (2 * a + 1))

def S(V, dB, S0, dt):
    """
    rBergomi price process.
    """
    # Construct non-anticipative Riemann increments
    increments = jnp.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = jnp.cumsum(increments, axis=1)

    S = jnp.zeros_like(V)
    S = S.at[:, 0].set(S0)
    S = S.at[:, 1:].set(S0 * jnp.exp(integral))
    return S

def S1(V, dW1, rho, S0, dt):
    """
    rBergomi parallel price process.
    """

    # Construct non-anticipative Riemann increments
    increments = rho * jnp.sqrt(V[:,:-1]) * dW1[:,:,0] - 0.5 * rho**2 * V[:,:-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = jnp.cumsum(increments, axis = 1)

    S = jnp.zeros_like(V)
    S = S.at[:, 0].set(S0)
    S = S.at[:, 1:].set(S0 * jnp.exp(integral))
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
    Covariance matrix for given alpha and n, assuming kappa = 1 for tractability.
    """
    cov = jnp.array([[0., 0.], [0., 0.]])
    cov = cov.at[0, 0].set(1. / n)
    cov = cov.at[0, 1].set(1. / ((a + 1) * n**(a + 1)))
    cov = cov.at[1, 1].set(1. / ((2. * a + 1) * n**(2. * a + 1)))
    cov = cov.at[1, 0].set(cov[0, 1])
    return cov

def fft_convolve(G, X):
    '''
    Fast Fourier transform version of convolve, 
    direct convolution is O(n^2) and FFT is O(n log n), so this implementation is faster for larger n values.'''
    # Zero-padding to the appropriate length for convolution
    G_padded = jnp.pad(G, (0, len(X) - 1))
    X_padded = jnp.pad(X, (0, len(G) - 1))

    # Perform FFT on both inputs
    G_fft = jnp.fft.fft(G_padded)
    X_fft = jnp.fft.fft(X_padded)

    # Element-wise multiplication in the Fourier domain
    result_fft = G_fft * X_fft

    # Inverse FFT to obtain the convolved result
    result = jnp.fft.ifft(result_fft)

    # Return the real part of the convolution
    return jnp.real(result[:len(X) + len(G) - 1])

@jit
def bs(F, K, V):
    '''
    Simple Black pricer for an OTM vol surface.
    Guards are now in place.
    '''
    eps = jnp.finfo(jnp.float32).eps

    w = 2 * (K > 1.) - 1  # 1 for call, -1 for put
    K = jnp.where(K == F, K + eps, K)  # Guard against K == F causing division to fail
    sv = jnp.sqrt(jnp.maximum(V, eps))  # Guard against V < 0 causing either sqrt or division to fail
    d1 = jnp.log(F / K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

@jit
def bs_obj(s, F, K, t, price): return bs(F=F, K=K, V=s**2*t) - price

@jit
def bs_inv(P, F, K, t, tol=1e-4, max_iter=1000):
    s_init = 0.2
    price = jnp.maximum(P, jnp.maximum((2 * (K > 1.0) - 1) * (F - K), jnp.finfo(jnp.float32).eps))
    bisect = jo.Bisection(bs_obj, 1e-7, 1e+7, tol=tol, maxiter=max_iter, check_bracket=False)
    s = bisect.run(s_init, F=F, K=K, t=t, price=price).params
    return s

# ---------------------------------------------------------
# Legacy functions
# ---------------------------------------------------------

def Y_legacy(N, s, n, a, dW):
    """
    Constructs Volterra process from appropriately correlated 2d Brownian increments.
    This version uses direct convolution which is O(n^2), which is slower than the FFT variant (O(n log n)) 
    so this version has been superseded.
    """
    Y1 = jnp.zeros((N, 1 + s)) # Exact integrals
    def y1(i, Y1): # Construct Y1 through exact integral
        Y1 = Y1.at[:, i].set(dW[:, i-1, 1]) # Assuming kappa = 1
        return Y1
    
    Y1 = lax.fori_loop(1, 1 + s, y1, Y1)
    Y2 = jnp.zeros((N, 1 + s)) # Riemann sums

    # Construct arrays for convolution
    G = jnp.zeros(1 + s) # Gamma
    def pre_generate_array(k, G):
        G = G.at[k].set(g(b(k, a) / n, a))
        return G

    G = lax.fori_loop(2, 1 + s, pre_generate_array, G)
    X = dW[:, :, 0]  # Xi

    # Compute convolution, FFT not used for small n
    # Possible to compute for all paths in C-layer?
    GX = jnp.zeros((N, len(X[0,:]) + len(G) - 1))
    def convolve_row(i, GX):
        GX = GX.at[i,:].set(jnp.convolve(G, X[i, :]))
        return GX
    
    GX = lax.fori_loop(0, N, convolve_row, GX)

    # Extract appropriate part of convolution
    Y2 = GX[:, :1 + s]

    # Finally, construct and return the full process
    Y = jnp.sqrt(2 * a + 1) * (Y1 + Y2)
    return Y