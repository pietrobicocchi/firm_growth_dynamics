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

# Spectral distance metric
def spectral_distance_metric(V0, V1, n=10):
    """
    Compute the spectral distance metric between the first n eigenvectors
    of two sets of eigenvectors stored in numpy arrays.
    cfr: J.P. sBouchaud
    Parameters:
    V0 (numpy array): The first set of eigenvectors.
    V1 (numpy array): The second set of eigenvectors.
    n (int): The number of eigenvectors to compare.

    Returns:
    dist (float): The spectral distance metric between the two sets of eigenvectors.
    """

    # Select the first n eigenvectors from the two sets
    V0 = V0[:, -n:]
    V1 = V1[:, -n:]
    # V1 = V1[:, :n] random overlap

    # Compute the matrix product of the two sets of eigenvectors
    A = V0.T @ V1

    # Compute the singular values of the matrix product
    s = np.linalg.svd(A, compute_uv=False)

    # Compute the spectral distance metric
    p = n
    dist = -np.sum(np.log(s)) / (2 * p + 1)

    return dist


# Density of an eigenvector
def density_eigenvector_metric(eigenvectors, number_eVec=10):
    """
    Compute the density eigenvector metric for the top eigenvectors.

    The density eigenvector metric is calculated as the sum of the fourth power of the elements in the top eigenvectors.

    Args:
        eigenvectors (numpy.ndarray): Array of eigenvectors.
        number_eVec (int, optional): Number of top eigenvectors to consider. Defaults to 10.

    Returns:
        numpy.ndarray: Array of density eigenvector metrics.

    Raises:
        ValueError: If the specified number of eigenvectors is greater than the available eigenvectors.

    Examples:
        >>> eigenvectors = np.array([[0.2, 0.3, 0.4], [0.1, 0.5, 0.6]])
        >>> density_eigenvector_metric(eigenvectors, number_eVec=2)
        array([0.5, 0.3])
    """

    if number_eVec > eigenvectors.shape[0]:
        raise ValueError("The specified number of eigenvectors is greater than the available eigenvectors.")

    top_eigenvectors = eigenvectors[-int(number_eVec):]
    metric = np.sum(top_eigenvectors ** 4, axis=1)
    metric = metric[::-1]

    return metric


# Mode coefficient
def mode_coefficient(returns, mode):
    """
    Compute the mode coefficient for a given mode λ using Principal Component Analysis (PCA).

    The mode coefficient represents the projection of the returns onto the eigenvector corresponding to the given mode.

    Args:
        returns (numpy.ndarray or pandas.DataFrame): Array-like object containing the returns data.
        mode (int): Index of the mode for which to compute the coefficient.

    Returns:
        numpy.ndarray: Array of mode coefficients.

    Raises:
        ValueError: If the specified mode is greater than the number of available modes.

    Examples:
        >>> returns = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> mode_coefficient(returns, mode=1)
        array([0.18569534, 0.23299483])
    """

    corr = returns.corr()
    # Perform PCA on the return data to obtain the eigenvectors and eigenvalues
    pca = PCA().fit(corr)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    if mode >= len(eigenvectors):
        raise ValueError("The specified mode is greater than the number of available modes.")

    # Compute the mode coefficient for the given mode λ
    mode_vector = eigenvectors[mode]
    mode_coeff = np.dot(mode_vector, returns.T)

    return mode_coeff


def analyze_mode_coefficients(returns):
    """
    Perform Fourier analysis on mode coefficients obtained from PCA.

    The function computes the mode coefficients using the `mode_coefficient` function for each mode. It then performs
    Fourier analysis on the mode coefficients to determine the periodicity and power spectrum of each mode.

    Args:
        returns (numpy.ndarray or pandas.DataFrame): Array-like object containing the returns data.

    Returns:
        list: List of periods (in days) for each mode.
        list: List of power spectra for each mode.

    Examples:
        >>> returns = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> periods, power_spectra = analyze_mode_coefficients(returns)
        >>> len(periods)
        2
        >>> len(power_spectra)
        2
    """

    mode_coeffs = []
    for i in range(returns.shape[1]):
        mode_coeffs.append(mode_coefficient(returns, i))

    N = len(mode_coeffs[0])
    periods = []
    power_spectra = []

    for mode_coeff in mode_coeffs:
        y = mode_coeff - np.mean(mode_coeff)  # remove DC component
        fft_y = np.fft.fft(y)
        freqs = np.fft.fftfreq(N)
        idx = np.argmax(np.abs(fft_y[1:N // 2])) + 1
        periods.append(1 / freqs[idx])
        power_spectrum = np.abs(fft_y) ** 2
        power_spectra.append(power_spectrum[1:N // 2])

    return periods, power_spectra


def plot_mode_analysis(periods, power_spectra, returns):
    """
    Plot the results of mode coefficient analysis.

    The function plots the periodicity and power spectrum of the mode coefficients.

    Args:
        periods (list): List of periods (in days) for each mode.
        power_spectra (list): List of power spectra for each mode.

    Returns:
        None

    Examples:
        >>> periods = [10, 20, 30]
        >>> power_spectra = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> plot_mode_analysis(periods, power_spectra)
        (plot is displayed)
    """
    T = returns.shape[0]
    np.fft.fftfreq(T)[1:T//2]
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].bar(range(len(periods)), periods, width=5)
    axs[0].set_xlabel('Mode')
    axs[0].set_ylabel('Period (days)')
    axs[0].set_title('Periodicity of mode coefficients')
    axs[1].plot(np.fft.fftfreq(T)[1:T//2], np.array(power_spectra).T)
    axs[1].set_xlabel('Frequency (1/days)')
    axs[1].set_ylabel('Power spectrum')
    axs[1].set_title('Power spectrum of mode coefficients')

    return fig


def plot_by_sector(mode, eVec, tk_ind):
    idx_by_sector, sector_dict, sector_indices, sector_labels = create_sectors(tk_ind)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(eVec[idx_by_sector, int(mode)].flatten())
    ax.axhline(0, c="green")
    # ax.set_ylim(-0.05, 0.2)
    # set x-axis labels
    ax.set_xticks(sector_indices)
    ax.set_xticklabels(sector_labels, rotation=45)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'brown', 'pink', 'gray', 'black',
              'cyan']  # list of colors
    color_idx = 0

    # plot vertical lines at sector boundaries
    prev_sector = tk_ind[idx_by_sector[0]]
    for i in range(1, len(idx_by_sector)):
        curr_sector = tk_ind[idx_by_sector[i]]
        if curr_sector != prev_sector:
            ax.axvline(i - 0.5, linestyle='-', color='grey')
            ax.fill_between([i - 1, i], -0.1, 0.1, color=colors[color_idx], alpha=0.4)
            # ax.fill_between([i-1, i], ax.get_ylim()[0], ax.get_ylim()[1], color=colors[color_idx], alpha=0.1)
            color_idx = (color_idx + 1) % len(colors)  # cycle through the list of colors

            prev_sector = curr_sector

    plt.title("plotting the {0} mode".format(mode))
    plt.show()


# Lead-Lag effects
def lagged_correlation(returns, tau=0):
    '''
    Lagged correlation matrix with shift = τ

    Parametres:
        - returns = dataframe pandas with all the rescaled log-returns of the stocks
        - tau = lagging time
    Returns:
        cross correlation matrix

    '''
    cross_corr_matrix = pd.DataFrame(columns=returns.columns, index=returns.columns)  # initialize the matrix

    for i in range(len(returns.columns)):
        for j in range(len(returns.columns)):
            cross_corr_matrix.iloc[i, j] = round(
                np.correlate(returns.iloc[:, i], returns.iloc[tau:, j], mode='valid')[0] / (len(returns) - tau),
                4)  # definition of correlation
    return cross_corr_matrix


def double_average(sales, prices, tau):
    corr_series = pd.Series(index=sales.columns)  # initialize the series
    for i in range(len(sales.columns)):
        corr_series.iloc[i] = round(
            np.correlate(sales.iloc[:, i], prices.iloc[tau:, i], mode='valid')[0] / (len(sales) - tau), 4)
    mean = corr_series.mean()

    return mean


def double_average_lags(periods, return_1, return_2):
    double_avgs = []
    for t in periods:
        double_avgs.append(double_average(return_1, return_2, t))
    return double_avgs


def lagged_correlation_mixed(ret1, ret2, tau=0):
    '''
    Lagged correlation matrix with shift = τ

    Parametres:
        - ret1
        - ret2
        - tau = lagging time
    Returns:
        cross correlation matrix between ret1 & ret2 with time-lag tau

    '''
    cross_corr_matrix = pd.DataFrame(columns=ret1.columns, index=ret1.columns)  # initialize the matrix

    for i in range(len(ret1.columns)):
        for j in range(len(ret1.columns)):
            cross_corr_matrix.iloc[i, j] = round(
                np.correlate(ret1.iloc[:, i], ret2.iloc[tau:, j], mode='valid')[0] / (len(ret1) - tau),
                4)  # definition of correlation
    return cross_corr_matrix