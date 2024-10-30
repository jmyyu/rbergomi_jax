import jax.numpy as jnp
from jax import jit, vmap, lax, Array, debug
import src.core.forecasting.rough_bergomi.jax_rbergomi as jrb

@jit
def rbergomi_pricer_base_jit(key, H, eta, rho, xi, tau_option, k):

    MC_samples = 60_000     # CI calculation for this method suggested >= 46610 paths 
                            # for a 95% CI of a rel. error <= 0.001 (0.1%) at tau_option = 1
                            # Check accompanying Jupyter notebook for details.

                            # Additionally, Zeron & Ruiz (2020) used 60000 paths;
                            # wanted to replicate this here (also happens to be greater than
                            # the value suggested by 95% CI)

    a = H-0.5  # Alpha
    n = 365  # Number of time steps per year

    tau_max = 1  # Max time to expiry
    s_max = tau_max*n  # Max step size

    s_tau: Array = jnp.ceil(n*tau_option).astype(int)  # Steps for option's maturity, expiry-specific

    t: Array = jnp.linspace(0, tau_max, 1+s_max)[jnp.newaxis, :]  # Time grid for the option
    dt: float = 1./n  # Time steps

    e = jnp.array([0,0])  # Expectation vector for the multivariate normal
    c = jrb.cov(a, n)  # Covariance matrix

    # Generate random increments
    dW1, key1 = jrb.dW1(key, e, c, MC_samples, s_max)  # Generate random increments
    dW2, _ = jrb.dW2(key1, MC_samples, s_max, dt)  # Generate orthogonal increments

    Y = jrb.Y(MC_samples, s_max, n, a, dW1)  # Generate Volterra process 
    dB = jrb.dB(dW1, dW2, rho)  # Generate correlated Brownian motion increments

    V = jrb.V(Y, xi, eta, t, a)  # Compute the variance process paths
    S = jrb.S(V, dB, 1., dt)  # Get corresponding price paths
    ST = S[:, s_tau][:,jnp.newaxis]  # Terminal futures price S1T

    K = jnp.exp(k)  # Convert log-moneyness to moneyness

    payoffs = jnp.where(K > 1, jnp.maximum(ST - K, 0), jnp.maximum(K - ST, 0))  # Get payoffs for each path
    price_raw = jnp.mean(payoffs, axis = 0)  # Take mean in order to get price (still needs cleaning)
    price_no_nans = jnp.where(jnp.isnan(price_raw), jnp.finfo(jnp.float32).eps, price_raw)  # Replace NaNs with eps
    price_final = jnp.maximum(price_no_nans, jnp.finfo(jnp.float32).eps)  # Replace values below eps with eps
    return vmap(jrb.bs_inv, in_axes=(0, None, 0, None))(price_final, 1., jnp.transpose(K), tau_option)

def vec_rbergomi_pricer(keys: Array, H: Array, eta: Array, rho: Array, xi: Array, tau_option: Array, k: Array):
    '''
    Computes volatility under rBergomi dynamics with MC sampling. 

    Code is now completely vectorized over every axes.
    
    IMPORTANT: Remember to split keys before use!
    As well, remember that all inputs should be JAX arrays.

    To change Monte Carlo simulation count and the time steps for the simulation, please directly modify in the source code;
    unfortunately, dynamically retrieving these quantities is not a feature that is compatible with JAX's JIT compilation.
    
    Args:
        keys: JAX keys for random number generation
        H: Hurst parameter
        eta: volatility of variance
        rho: correlation between stock and vol
        v0: spot variance
        tau_option: time to maturity of the option (in years)
        k: log-moneyness of options

    Returns:
        ivs: implied vols for all moneyness
    '''
    return vmap(rbergomi_pricer_base_jit, in_axes=(0, 0, 0, 0, 0, 0, 0))(keys, H, eta, rho, xi, tau_option, k)

@jit
def rbergomi_pricer_mixed_jit(key, H, eta, rho, xi, tau_option, k):

    MC_samples = 40_000     # CI calculation for this method suggested >= 37661 paths 
                            # for a 95% CI of a rel. error <= 0.001 (0.1%) at tau_option = 1
                            # Check accompanying Jupyter notebook for details.

    a = H-0.5  # Alpha
    n = 365  # Number of time steps per year
    
    tau_max = 1  # Max time to expiry
    s_max = tau_max*n  # Max step size
    
    s_tau: Array = jnp.ceil(n*tau_option).astype(int)  # Steps for option's maturity, expiry-specific

    t: Array = jnp.linspace(0, tau_max, 1+s_max)[jnp.newaxis, :]  # Time grid for the option
    dt: float = 1./n  # Time steps

    e: Array = jnp.array([0,0])  # Expectation vector for the multivariate normal
    c: Array = jrb.cov(a, n)  # Covariance matrix

    dW1, _ = jrb.dW1(key, e, c, MC_samples, s_max)  # Generate random increments
    Ya: Array = jrb.Y(MC_samples, s_max, n, a, dW1)  # Generate Volterra process 

    V: Array = jrb.V(Ya, xi, eta, t, a)  # Compute the variance process paths
    S1: Array = jrb.S1(V, dW1, rho, 1., dt)  # Get parallel price paths

    # Cut the processes down to size
    mask: Array = (jnp.arange(s_max + 1) < s_tau)[jnp.newaxis, :]
    V_masked: Array = jnp.where(mask, V, 0)  # Since downstream takes a sum value, zeroes won't impact anything

    QV: Array = jnp.sum(V_masked, axis = 1)[:,jnp.newaxis] * dt  # Quadratic variation
    Q: Array = jnp.max(QV) + jnp.finfo(jnp.float32).eps  # Variation budget

    K: Array = jnp.exp(k)  # Convert log-moneyness to moneyness

    S1T: Array = S1[:, s_tau][:,jnp.newaxis]  # Terminal futures price S1T
    X: Array = jrb.bs(S1T, K, (1 - rho**2) * QV)  # Black 76 driven by the uncorrelated part of vol proc (1 - rho^2)
    Y: Array = jrb.bs(S1T, K, rho**2 * (Q - QV))  # Black 76 driven by the correlated part (rho^2)
    eY: Array = jrb.bs(1., K, rho**2 * Q)  # Correction term based on the expectation of Y
    debug.print("X: {}, Y: {}, eY: {}", jnp.any(jnp.isnan(X)), jnp.any(jnp.isnan(Y)), jnp.any(jnp.isnan(eY)))

    c = jnp.zeros_like(k)[jnp.newaxis,:]
    def cov_mat_func(i, c):
        cov_mat = jnp.cov(X[:, i], Y[:, i])
        c = c.at[0, i].set(-cov_mat[0, 1] / cov_mat[1, 1])
        return c

    c = lax.fori_loop(0, len(k), cov_mat_func, c)  # Asymptotically optimal weights using covariance calculation
    debug.print('{}', jnp.isnan(c))
    payoffs: Array = X + c * (Y - eY)

    price_raw = jnp.mean(payoffs, axis = 0)  # Take mean in order to get price (still needs cleaning)
    price_no_nans = jnp.where(jnp.isnan(price_raw), 0., price_raw)  # Replace NaNs with 0
    price_final = jnp.maximum(price_no_nans, jnp.finfo(jnp.float32).eps)  # Replace values below eps with eps
    return vmap(jrb.bs_inv, in_axes=(0, None, 0, None))(price_final, 1., jnp.transpose(K), tau_option)

def vec_rbergomi_pricer_mixed(keys: Array, H: Array, eta: Array, rho: Array, xi: Array, tau_option: Array, k: Array):
    '''
    Computes volatility under rBergomi dynamics with MC sampling. 
    This uses the mixed estimator presented in McCrikerd & Pakkanen (2018) that can be directly used as a variance reduction method.

    Code is now completely vectorized over every axes.
    
    IMPORTANT: Remember to split keys before use!
    As well, remember that all inputs should be JAX arrays.

    To change Monte Carlo simulation count and the time steps for the simulation, please directly modify in the source code;
    unfortunately, dynamically retrieving these quantities is not a feature that is compatible with JAX's JIT compilation.
    
    Args:
        keys: JAX keys for random number generation
        H: Hurst parameter
        eta: volatility of variance
        rho: correlation between stock and vol
        v0: spot variance
        tau_option: time to maturity of the option (in years)
        k: log-moneyness of options

    Returns:
        ivs: implied vols for all moneyness
    '''
    return vmap(rbergomi_pricer_mixed_jit, in_axes=(0, 0, 0, 0, 0, 0, 0))(keys, H, eta, rho, xi, tau_option, k)


# ---------------------------------------------------------
# Vectorization attempt on tau_option...
# ---------------------------------------------------------

@jit
def rbergomi_pricer_base_jit_over_tau_option_k(key, H, eta, rho, xi, tau_option, k):

    MC_samples = 50_000     # CI calculation for this method suggested >= 46610 paths 
                            # for a 95% CI of a rel. error <= 0.001 (0.1%) at tau_option = 1
                            # Check accompanying Jupyter notebook for details.

    a = H-0.5  # Alpha
    n = 365  # Number of time steps per year

    tau_max = 1  # Max time to expiry
    s_max = tau_max*n  # Max step size

    s_tau: Array = jnp.ceil(n*tau_option).astype(int)  # Steps for option's maturity, expiry-specific

    t: Array = jnp.linspace(0, tau_max, 1+s_max)[jnp.newaxis, :]  # Time grid for the option
    dt: float = 1./n  # Time steps

    e = jnp.array([0,0])  # Expectation vector for the multivariate normal
    c = jrb.cov(a, n)  # Covariance matrix

    # Generate random increments
    dW1, key1 = jrb.dW1(key, e, c, MC_samples, s_max)  # Generate random increments
    dW2, _ = jrb.dW2(key1, MC_samples, s_max, dt)  # Generate orthogonal increments

    Y = jrb.Y(MC_samples, s_max, n, a, dW1)  # Generate Volterra process 
    dB = jrb.dB(dW1, dW2, rho)  # Generate correlated Brownian motion increments

    V = jrb.V(Y, xi, eta, t, a)  # Compute the variance process paths
    S = jrb.S(V, dB, 1., dt)  # Get corresponding price paths

    @jit
    def pricer(s_tau, k):
        ST = S[:, s_tau][:,jnp.newaxis]  # Terminal futures price S1T

        K = jnp.exp(k)  # Convert log-moneyness to moneyness

        payoffs = jnp.where(K > 1, jnp.maximum(ST - K, 0), jnp.maximum(K - ST, 0))  # Get payoffs for each path
        price_raw = jnp.mean(payoffs, axis = 0)  # Take mean in order to get price (still needs cleaning)
        price_no_nans = jnp.where(jnp.isnan(price_raw), jnp.finfo(jnp.float32).eps, price_raw)  # Replace NaNs with eps
        price_final = jnp.maximum(price_no_nans, jnp.finfo(jnp.float32).eps)  # Replace values below eps with eps
        return vmap(jrb.bs_inv, in_axes=(0, None, 0, None))(price_final, 1., jnp.transpose(K), tau_option)
    
    return vmap(pricer, in_axes=(0, 0))(s_tau, k)

def vec_rbergomi_pricer_tau(keys: Array, H: Array, eta: Array, rho: Array, xi: Array, tau_option: Array, k: Array):
    return vmap(rbergomi_pricer_mixed_jit_over_tau_option_k, in_axes=(0, 0, 0, 0, 0, 0, 0))(keys, H, eta, rho, xi, tau_option, k)

@jit
def rbergomi_pricer_mixed_jit_over_tau_option_k(key, H, eta, rho, xi, tau_option, k):

    MC_samples = 40_000     # CI calculation for this method suggested >= 37661 paths 
                            # for a 95% CI of a rel. error <= 0.001 (0.1%) at tau_option = 1
                            # Check accompanying Jupyter notebook for details.

    a = H-0.5  # Alpha
    n = 365  # Number of time steps per year
    
    tau_max = 1  # Max time to expiry
    s_max = tau_max*n  # Max step size
    
    s_tau: Array = jnp.ceil(n*tau_option).astype(int)  # Steps for option's maturity, expiry-specific

    t: Array = jnp.linspace(0, tau_max, 1+s_max)[jnp.newaxis, :]  # Time grid for the option
    dt: float = 1./n  # Time steps

    e: Array = jnp.array([0,0])  # Expectation vector for the multivariate normal
    c: Array = jrb.cov(a, n)  # Covariance matrix

    dW1, _ = jrb.dW1(key, e, c, MC_samples, s_max)  # Generate random increments
    Ya: Array = jrb.Y(MC_samples, s_max, n, a, dW1)  # Generate Volterra process 

    V: Array = jrb.V(Ya, xi, eta, t, a)  # Compute the variance process paths
    S1: Array = jrb.S1(V, dW1, rho, 1., dt)  # Get parallel price paths

    @jit
    def pricer(s_tau, k):
        # Cut the processes down to size
        mask: Array = (jnp.arange(s_max + 1) < s_tau)[jnp.newaxis, :]
        V_masked: Array = jnp.where(mask, V, 0)  # Since downstream takes a sum value, zeroes won't impact anything

        QV: Array = jnp.sum(V_masked, axis = 1)[:,jnp.newaxis] * dt  # Quadratic variation
        Q: Array = jnp.max(QV) + jnp.finfo(jnp.float32).eps  # Variation budget

        K: Array = jnp.exp(k)[jnp.newaxis,:]  # Convert log-moneyness to moneyness

        S1T: Array = S1[:, s_tau][:,jnp.newaxis]  # Terminal futures price S1T
        X: Array = jrb.bs(S1T, K, (1 - rho**2) * QV)  # Black 76 driven by the uncorrelated part of vol proc (1 - rho^2)
        Y: Array = jrb.bs(S1T, K, rho**2 * (Q - QV))  # Black 76 driven by the correlated part (rho^2)
        eY: Array = jrb.bs(1., K, rho**2 * Q)  # Correction term based on the expectation of Y

        c = jnp.zeros_like(k)[jnp.newaxis,:]
        def cov_mat_func(i, c):
            cov_mat = jnp.cov(X[:, i], Y[:, i])
            c = c.at[0, i].set(-cov_mat[0, 1] / cov_mat[1, 1])
            return c

        c = lax.fori_loop(0, len(k), cov_mat_func, c)  # Asymptotically optimal weights using covariance calculation
        mixed_payoffs: Array = X + c * (Y - eY)
        mixed_prices_raw: Array = jnp.mean(mixed_payoffs, axis = 0)
        mixed_prices_no_nans = jnp.where(jnp.isnan(mixed_prices_raw), jnp.finfo(jnp.float32).eps, mixed_prices_raw)  # Replace NaNs with eps
        mixed_prices_final = jnp.maximum(mixed_prices_no_nans, jnp.finfo(jnp.float32).eps)[:, jnp.newaxis]  # Replace values below eps with eps
        return jrb.bs_inv(mixed_prices_final, 1., jnp.transpose(K), tau_option)

    return vmap(pricer, in_axes=(0, 0))(s_tau, k)

def vec_rbergomi_pricer_mixed_tau(keys: Array, H: Array, eta: Array, rho: Array, xi: Array, tau_option: Array, k: Array):
    return vmap(rbergomi_pricer_mixed_jit_over_tau_option_k, in_axes=(0, 0, 0, 0, 0, 0, 0))(keys, H, eta, rho, xi, tau_option, k)