import numpy as np
from src.core.forecasting.rough_bergomi.np_rbergomi_oop.rbergomi import rBergomi
from src.core.forecasting.rough_bergomi.np_rbergomi_oop.utils import bs, bsinv

def turbocharged(H, eta, rho, xi, tau_option, k, MC_samples):
    vec_bsinv = np.vectorize(bsinv)

    rB = rBergomi(n = 365, N = MC_samples, T = tau_option, a = H-0.5)
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    Ya = rB.Y(dW1)
    dB = rB.dB(dW1, dW2, rho)
    V = rB.V(Ya, xi, eta) 
    S = rB.S(V, dB)
    K = np.exp(k)[np.newaxis,:]
    ST = S[:,-1][:,np.newaxis]
    base_payoffs = np.maximum(ST - K,0)
    base_prices = np.mean(base_payoffs, axis = 0)[:,np.newaxis]
    base_vols = vec_bsinv(base_prices, 1., np.transpose(K), rB.T)

    return base_vols

def turbocharged_mixed(H, eta, rho, xi, tau_option, k, MC_samples):
    vec_bsinv = np.vectorize(bsinv)

    rB = rBergomi(n = 365, N = MC_samples, T = tau_option, a = H-0.5)
    dW1 = rB.dW1()
    Ya = rB.Y(dW1)
    V = rB.V(Ya, xi, eta) 
    S1 = rB.S1(V, dW1, rho)
    K = np.exp(k)[np.newaxis,:]
    QV = np.sum(V, axis = 1)[:,np.newaxis] * rB.dt
    Q = np.max(QV) + 1e-9 
    S1T = S1[:,-1][:,np.newaxis]
    X = bs(S1T, K, (1 - rho**2) * QV)
    Y = bs(S1T, K, rho**2 * (Q - QV))
    eY = bs(1., K, rho**2 * Q)

    # Asymptotically optimal weights
    c = np.zeros_like(k)[np.newaxis,:]
    for i in range(len(k)):
        cov_mat = np.cov(X[:,i], Y[:,i])
        c[0,i] = - cov_mat[0,1] / cov_mat[1,1]

    # Payoffs, prices and volatilities
    mixed_payoffs = X + c * (Y - eY)
    mixed_prices = np.mean(mixed_payoffs, axis = 0)[:,np.newaxis]
    mixed_vols = vec_bsinv(mixed_prices, 1., np.transpose(K), rB.T)
    return mixed_vols