import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from arch import arch_model

from src.csv_data.utils import *

cwd = os.getcwd()  # working directory
parent_dir = os.path.dirname(cwd)
env_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path=env_path)  # take environment variables from .env.

DATA_DIR = os.getenv('DATA_DIR')
USER_PATH = os.getenv('USER_PATH')


def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.

    Parameters:
        matrix (numpy.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    # Eigenvalues test
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return True

    # Cholesky decomposition test
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

    return eigenvalues


def multivariate_rand_samp(corr, n_samples=300):
    """
    Generate multivariate random samples with a specified correlation structure.

    Parameters:
        corr (numpy.ndarray): The correlation matrix representing the desired correlation structure.
        n_samples (int): Number of random samples to generate (default: 300).

    Returns:
        pandas.DataFrame: A DataFrame containing the generated multivariate random samples.

    Notes:
        This function uses the Cholesky decomposition method to generate random samples with the desired
        correlation structure. The generated samples are drawn from a multivariate normal distribution.

    Example:
        # Generate 500 random samples with a correlation matrix 'corr'
        samples = multivariate_rand_samp(corr, n_samples=500)
    """

    # Cholesky decomposition
    lower_triangle = np.linalg.cholesky(corr)

    # Dimensionality of each sample
    dimension = corr.shape[0]

    # Generate random Gaussian samples with the desired correlation
    samples = np.random.randn(n_samples, dimension) @ lower_triangle

    # Convert the samples to a pandas DataFrame
    samples = pd.DataFrame(samples)

    return samples


def generate_garch_synthetic_returns(returns, p=1, q=1, n_samples=300):
    """
    Generate synthetic returns time series with autocorrelation and volatility clustering using a GARCH model.

    Parameters:
        returns (pandas.DataFrame): Historical returns data for multiple companies. Columns represent different companies.
        p (int): Order of the autoregressive component of the GARCH model (default: 1).
        q (int): Order of the moving average component of the GARCH model (default: 1).
        n_samples (int): Number of synthetic returns time series to generate (default: 300).

    Returns:
        pandas.DataFrame: A DataFrame containing the generated synthetic returns time series for each company.

    Notes:
        This function fits a separate GARCH(p, q) model to the historical returns data for each company and uses
        each model to generate synthetic returns time series that exhibit autocorrelation and volatility clustering.

    Example:
        # Generate 500 synthetic returns time series using GARCH(1, 1) model on historical returns data 'returns'
        synthetic_returns = generate_garch_synthetic_returns(returns, p=1, q=1, n_samples=500)
    """

    num_companies = returns.shape[1]
    synthetic_returns = pd.DataFrame()

    for i in range(num_companies):
        # Extract returns for a single company
        company_returns = returns.iloc[:, i]

        # Fit GARCH model for the company
        model = arch_model(company_returns, vol='Garch', p=p, q=q)
        model_fit = model.fit(disp='off')

        # Generate synthetic returns for the company
        synthetic_company_returns = model_fit.forecast(start=model_fit.conditional_volatility.shape[0], horizon=n_samples)
        synthetic_company_returns = synthetic_company_returns.mean.values[-1, :]

        # Append synthetic returns to the DataFrame
        synthetic_returns[returns.columns[i]] = synthetic_company_returns

    return synthetic_returns


def mode_coefficient_matrix(returns):
    '''
    computing the projection onto the eigenspace of the returns
    Return:
        mode_coeffs, each column j correspond to the importance of mode j at each time step
    '''
    corr = returns.corr()

    # Perform Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(corr)

    mode_coeffs = np.dot(returns, eigenvectors)

    return mode_coeffs


def plot_distribution(time_series):
    # Calculate the histogram of the time series
    hist, bins = np.histogram(time_series, bins=80)

    # Calculate the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the distribution
    plt.figure()
    plt.bar(bin_centers, hist, width=(bin_centers[1] - bin_centers[0]), align='center')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time Series Values')
    plt.show()


def generate_sample_time_series(time_series):
    # Compute the distribution function of the time series
    values, counts = np.unique(time_series, return_counts=True)
    distribution = np.cumsum(counts) / len(time_series)

    # Generate random samples from the distribution
    random_samples = np.random.random(len(time_series))

    # Map the random samples to the original time series distribution
    sampled_time_series = np.interp(random_samples, distribution, values)

    return sampled_time_series


def synthetic_returns_shuffling(returns, eigenvectors):
    # Project onto Eigenspace
    Y = mode_coefficient_matrix(returns)

    # Shuffle Projection Components
    Y_shuffle = Y.copy()  # Copy the projected matrix
    np.random.shuffle(Y_shuffle)  # Shuffle the rows of the projected matrix

    # Generate Synthetic Time Series
    X_synthetic = np.dot(Y_shuffle, np.linalg.inv(eigenvectors))

    return X_synthetic


def synthetic_returns_sampling(returns, eigenvectors):
    # Project onto Eigenspace
    Y = mode_coefficient_matrix(returns)
    Y_sample = {}

    # Sample from distribution
    for i in range(len(returns.keys())):
        sy = generate_sample_time_series(Y[:, i])
        Y_sample[i] = sy

    Y_sample = pd.DataFrame(Y_sample)

    # Generate Synthetic Time Series
    X_synthetic = np.dot(Y_sample, np.linalg.inv(eigenvectors))

    return X_synthetic


def tailored_clean(sales, sales_keys):
    sales = sales.replace(0, np.nan)

    def drop_companies(df, keys):
        return df.loc[:, df.columns.intersection(keys)]

    sales = drop_companies(df=sales, keys=sales_keys)  # Dropping companies
    sales = sales.fillna(0)

    log_ret = np.log(sales).diff()
    log_ret.replace([np.inf, -np.inf], np.nan, inplace=True)

    resc_log_ret = (log_ret - log_ret.mean()) / log_ret.std()
    resc_log_ret = resc_log_ret.fillna(0)

    return resc_log_ret