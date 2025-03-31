#%%
#ORIGINALE DI FABIO SU CHIMERE2


import numpy as np
import pandas as pd
import torch  # PyTorch for CUDA acceleration
import torch.multiprocessing as mp
from scipy.optimize import newton
from scipy.stats import norm
from scipy.optimize import brentq
import time
import matplotlib.pyplot as plt
#from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
# Define function to simulate SABR model using GPU (PyTorch)
import torch
torch.backends.cuda.max_split_size_mb = 128

def sabr_monte_carlo(F0, T, alpha, beta, rho, nu, r, N_paths, N_steps):
    """
    SABR Monte Carlo simulation for option pricing.

    Parameters:
    - F0: Initial forward price
    - T: Time to maturity
    - alpha: Initial volatility (sigma_0)
    - beta: Elasticity parameter
    - rho: Correlation between forward and volatility
    - nu: Volatility of volatility
    - r: Risk-free rate
    - n_paths: Number of Monte Carlo paths
    - n_steps: Number of time steps for the simulation

    Returns:
    - Simulated forward prices (2D array: n_paths x n_steps + 1)
    - Simulated volatilities (2D array: n_paths x n_steps + 1)
    """
    dt = 1 / N_steps  # Time step size
    
    # Initialize forward prices and volatilities
    S = np.full((N_paths, int(T * N_steps + 1)), F0)  # Forward prices (n_paths x n_steps + 1)
    sigma = np.full((N_paths, int(T * N_steps + 1)), alpha)  # Volatilities (n_paths x n_steps + 1)
    
    # Correlation matrix and Cholesky decomposition
    corr_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(corr_matrix)  # Cholesky decomposition
    
    # Time stepping
    for step in range(int(T * N_steps)):
        # Generate independent standard normal random variables
        Z = np.random.normal(0, 1, size=(2, N_paths))  # Shape (2, n_paths)
        
        # Correlate the random variables using the Cholesky decomposition
        correlated_Z = L @ Z  # Shape (2, n_paths)
        dW = np.sqrt(dt) * correlated_Z[0]  # Correlated Brownian motion for dW_t
        dZ = np.sqrt(dt) * correlated_Z[1]  # Correlated Brownian motion for dZ_t
        
        # Update volatility process
        sigma[:, step + 1] = sigma[:, step] * np.exp(nu * dZ - 0.5 * nu**2 * dt)
        
        # Update forward price process
        S[:, step + 1] = S[:, step] + sigma[:, step] * (S[:, step] ** beta) * dW
        
    return S, sigma

# Function to compute option price
def compute_option_price(S, K, r, T):
    payoff = np.maximum(S - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

# Black-Scholes function for implied volatility calculation
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Function to compute implied volatility using Brent's method
def implied_volatility(price, S0, K, T, r,sigma):
    func = lambda sigma: black_scholes_call(S0, K, T, r, sigma) - price
    try:
        return brentq(func, 0.001, 5.0)
    except ValueError:
        return np.nan
    
def sabr_implied_volatility(alpha, beta, rho, nu, F, K, T):
    if F == K:
        FK_avg = F
    else:
        FK_avg = np.sqrt(F * K)  # Geometric mean
    
    # Log-moneyness
    log_FK = np.log(F / K)
    
    # Compute z and D(z)
    z = (nu / alpha) * (FK_avg ** ((1 - beta) / 2)) * log_FK
    D_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    
    # A term (Numerator adjustment)
    A = (1 + ((1 - beta)**2 / 24) * (alpha**2 / (FK_avg ** (2 * (1 - beta)))) +
         (rho * beta * nu / 4) * (alpha / (FK_avg ** (1 - beta))) +
         ((2 - 3 * rho**2) / 24) * (nu**2))
    
    # B term (Denominator adjustment)
    B = (1 + ((1 - beta)**2 / 24) * (log_FK)**2 +
         ((1 - beta)**4 / 1920) * (log_FK)**4)
    
    # SABR implied volatility approximation
    sigma_impl = (alpha / (FK_avg ** ((1 - beta) / 2))) * (A / B) * (z / D_z)
    
    return sigma_impl

# Function for processing each parameter sample (to run in parallel)
def process_sample(sample):
    S0, alpha, beta, rho, nu = sample
    r = np.random.uniform(0.01, 0.05)
    
    maturities = np.arange(0.2, 2.1, 0.3).round(1).tolist()
    strikes = np.arange(50, 155, 5).tolist()
    results = []

    for T in maturities:

        S,sigma = sabr_monte_carlo(S0*np.exp(r*T), T, alpha, beta, rho, nu, r, N_paths, N_steps)
        
        for K in strikes:
            real = sabr_implied_volatility(alpha, beta, rho, nu, S0*np.exp(r*T), K, T)
            SABR_option_price = compute_option_price(S[:, -1], K, r, T)
            iv= implied_volatility(SABR_option_price, S0, K, T, r, sigma)

            if not np.isnan(iv) and iv != 0.001:

                bs = black_scholes_call(S0, K, T, r, sigma[0][0])
                bs_IV = black_scholes_call(S0, K, T, r, iv)
                results.append([S0, alpha, beta, rho, nu, r, T, K, iv,real, bs, SABR_option_price, bs_IV])

    return results

# Define ranges for random parameter sampling
S0_range = (80, 130)
beta_range = (1, 1)
nu_range = (0.05, 0.3)

N_paths = 60000
N_steps = 252
N_samples = 30000 # Number of parameter samples forse 15 ore di esecuzione
#N_samples = 100 # Number of parameter samples forse 15 ore di esecuzione

# Generate random parameter samples
parameter_S0 = np.random.uniform(S0_range[0],S0_range[1],size=N_samples)
parameter_alpha = np.maximum(0.05, np.minimum(np.random.normal(0.2, 0.05,size=N_samples), 0.4))
parameter_beta = np.random.uniform(beta_range[0],beta_range[1],size=N_samples)
parameter_rho = np.maximum(-0.99, np.minimum(np.random.normal(0, 0.5,size=N_samples), 0.99))
parameter_nu = np.random.uniform(nu_range[0],nu_range[1],size=N_samples)

parameter_samples = np.column_stack((parameter_S0, parameter_alpha, parameter_beta, parameter_rho, parameter_nu))

print(parameter_samples.shape)
# Parallel execution using multiprocessing
if __name__ == "__main__":
    start_time = time.time()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_sample, parameter_samples)

    # Flatten results and convert to DataFrame
    flattened_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened_results, columns=["S0", "alpha", "beta", "rho", "nu", "r", "T", "K", "IV","IV_Approx", "BS Price", "SABR Price", "BS Price IV"])
    
    # Save results
    strCSV = 'SABR_model_DEF_PARTE02.csv'
    df.to_csv(strCSV, index=False)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
# %%
