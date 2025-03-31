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
    - N_paths: Number of Monte Carlo paths
    - N_steps: Number of time steps for the simulation

    Returns:
    - Simulated forward prices (2D array: N_paths x N_steps + 1)
    - Simulated volatilities (2D array: N_paths x N_steps + 1)
    """
    dt = 1 / N_steps  # Time step size
    
    # Initialize forward prices and volatilities
    S = np.full((N_paths, int(T * N_steps + 1)), F0)  # Forward prices (N_paths x N_steps + 1)
    sigma = np.full((N_paths, int(T * N_steps + 1)), alpha)  # Volatilities (N_paths x N_steps + 1)
    
    # Correlation matrix and Cholesky decomposition
    corr_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(corr_matrix)  # Cholesky decomposition
    
    # Time stepping
    for step in range(int(T * N_steps)):
        # Generate independent standard normal random variables
        Z = np.random.normal(0, 1, size=(2, N_paths))  # Shape (2, N_paths)
        
        # Correlate the random variables using the Cholesky decomposition
        correlated_Z = L @ Z  # Shape (2, N_paths)
        dW = np.sqrt(dt) * correlated_Z[0]  # Correlated Brownian motion for dW_t
        dZ = np.sqrt(dt) * correlated_Z[1]  # Correlated Brownian motion for dZ_t
        
        # Update volatility process
        sigma[:, step + 1] = sigma[:, step] * np.exp(nu * dZ - 0.5 * nu**2 * dt)
        
        # Update forward price process
        S[:, step + 1] = S[:, step] + sigma[:, step] * (S[:, step] ** beta) * dW
    
    return S, sigma

# Function for processing each parameter sample (to run in parallel)
def process_sample(sample):
    S0, alpha, beta, rho, nu, r = sample
  
    #maturities = np.arange(0.2, 2.1, 0.3).round(1).tolist()# 1.4
    #strikes = np.arange(50, 155, 5).tolist()
    results = []
    T = 1.4
    #for T in maturities:
    S, sigma = sabr_monte_carlo(S0*np.exp(r*T), T, alpha, beta, rho, nu, r, N_paths, N_steps)
    results.append(S)  # Add 2D array S (N_paths x N_steps)
    results.append(sigma)
    results.append([S0, alpha, beta, rho, nu, r, T])
        
    return results

# Define ranges for random parameter sampling
S0_range = (80, 130)
beta_range = (1, 1)
nu_range = (0.05, 0.3)

N_paths = 10000
N_steps = 252
N_samples = 1  # Number of parameter samples

# Generate random parameter samples
parameter_S0 = np.random.uniform(S0_range[0], S0_range[1], size=N_samples)
parameter_alpha = np.maximum(0.05, np.minimum(np.random.normal(0.2, 0.05, size=N_samples), 0.4))
parameter_beta = np.random.uniform(beta_range[0], beta_range[1], size=N_samples)
parameter_rho = np.maximum(-0.99, np.minimum(np.random.normal(0, 0.5, size=N_samples), 0.99))
parameter_nu = np.random.uniform(nu_range[0], nu_range[1], size=N_samples)
parameter_r = np.random.uniform(0.01, 0.05, size=N_samples)
parameter_samples = np.column_stack((parameter_S0, parameter_alpha, parameter_beta, parameter_rho, parameter_nu, parameter_r))


# Parallel execution using multiprocessing
if __name__ == "__main__":
    start_time = time.time()
    
    # Create a multiprocessing pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Run the function in parallel
        results = pool.map(process_sample, parameter_samples)
    flattened_S = []
    flattened_sigma = []
    parameters_list = []

    # Process each result for each maturity T
    for result in results:
        S = result[0]  # The forward prices matrix (N_paths x N_steps)
        sigma = result[1]  # The volatility matrix (N_paths x N_steps)
        parameters = result[2]  # Parameters list [S0, alpha, beta, rho, nu, r, T]
        
        # Flatten the S and sigma matrices row by row
        flattened_S.append(S)  # Do not flatten here yet
        flattened_sigma.append(sigma)  # Do not flatten here yet
        
        # Repeat the parameters for each path
        parameters_repeated = np.tile(parameters, (S.shape[0], 1))  # Repeat parameters N_paths times
        parameters_list.append(parameters_repeated)

    # Now combine the results row by row
    combined_data = []

    for i in range(len(flattened_S)):
        # Get the corresponding S and sigma for this maturity
        S_row = flattened_S[i]
        sigma_row = flattened_sigma[i]
        
        # Concatenate S_row and sigma_row row by row
        for j in range(S_row.shape[0]):  # Loop over each path
            combined_row = np.concatenate((S_row[j], sigma_row[j], parameters_list[i][j]))
            combined_data.append(combined_row)

    # Convert combined data into a DataFrame
    N_steps = S.shape[1]  # Assuming all S matrices have the same shape

# Generate column names dynamically based on N_steps
    columns = [f'S_{i+1}' for i in range(N_steps)] + [f'sigma_{i+1}' for i in range(N_steps)] + ['S0', 'alpha', 'beta', 'rho', 'nu', 'r', 'T']

    # Create DataFrame
    df = pd.DataFrame(combined_data, columns=columns)

    # Display the first few rows of the DataFrame to verify
    print(df.head())
    strCSV = 'SABR_model_results.csv'
    df.to_csv(strCSV, index=False)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    