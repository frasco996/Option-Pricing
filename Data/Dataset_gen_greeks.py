import pandas as pd
import numpy as np
import time
from scipy.stats import norm
from multiprocessing import Pool, cpu_count, Value, Lock
import os


# Shared counter for tracking processed rows
processed_rows = Value("i", 0)  # Shared integer value
lock = Lock()  # Ensures safe updates across processes

def calculate_greeks(row):
    """Calculates Delta, Vega, Theta, and Rho along with their computation times."""
    S0 = row['initial_stock_price']
    K = row['strike_price']
    T = row['time_to_maturity'] / 240  # Convert days to years
    r = row['interest_rate']
    sigma = row['volatility']
    
    if T == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Delta Calculation
    start_delta = time.time()
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    end_delta = time.time()
    delta_time = end_delta - start_delta

    # Vega Calculation
    start_vega = time.time()
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    end_vega = time.time()
    vega_time = end_vega - start_vega

    # Theta Calculation
    start_theta = time.time()
    d2 = d1 - sigma * np.sqrt(T)
    theta = (- (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    end_theta = time.time()
    theta_time = end_theta - start_theta

    # Rho Calculation
    start_rho = time.time()
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    end_rho = time.time()
    rho_time = end_rho - start_rho

    return delta, delta_time, vega, vega_time, theta, theta_time, rho, rho_time

def process_chunk(df_chunk):
    """Processes a chunk of the dataframe and prints progress every 1,000,000 rows."""
    results = []
    
    global processed_rows, lock
    for _, row in df_chunk.iterrows():
        results.append(calculate_greeks(row))

        with lock:  # Ensure safe update of shared counter
            processed_rows.value += 1
            if processed_rows.value % 1_000_000 == 0:
                print(f"Processed {processed_rows.value} rows...")

    return pd.DataFrame(results, columns=['Delta', 'Delta_time', 'Vega', 'Vega_time', 'Theta', 'Theta_time', 'Rho', 'Rho_time'], index=df_chunk.index)

def parallel_apply(df, func, num_partitions=None):
    if num_partitions is None:
        num_partitions = cpu_count()  # Use all available CPUs
    
    df_split = np.array_split(df, num_partitions)  # Split dataframe into chunks
    
    with Pool(num_partitions) as pool:
        results = pool.map(func, df_split)  # Process chunks in parallel
    
    return pd.concat(results)  # Merge results back

if __name__ == "__main__":
    print("Loading dataset...")
    option_price = pd.read_csv('./option_pricing_full_dataset.csv')
    print("Dataset Loaded")

    # Parallel processing
    print("Starting multiprocessing...")
    option_price[['Delta', 'Delta_time', 'Vega', 'Vega_time', 'Theta', 'Theta_time', 'Rho', 'Rho_time']] = parallel_apply(option_price, process_chunk)

    print(f"Processing complete. Total rows processed: {processed_rows.value}")
    print(option_price.head())


    output_folder = "./"
    output_file = output_folder + "option_pricing_with_greeks.csv"

    # Ensure the folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the DataFrame
    option_price.to_csv(output_file, index=False)
    print(f"Data saved successfully in: {output_file}")
    