import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def mpPDF(var, q, pts):
    """
    Calculate the Marčenko-Pastur probability density function (PDF) for random matrices.

    Args:
        var (float): The variance of the random matrix variable.
        q (float): The ratio of the number of observations (ex time) to the number of variables (q = N / M). (ex q = T(N)
        pts (int): The number of points at which to evaluate the PDF.

    Returns:
        pd.Series: The Marčenko-Pastur PDF, represented as a Pandas Series with the PDF values as the data and
                   the corresponding eigenvalues as the index.

    Example:
        var = 0.5
        q = 2.0
        pts = 100

        pdf = mpPDF(var, q, pts)
    """
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2  # calc lambda_minus, lambda_plus
    eVal = np.linspace(eMin, eMax, pts) #Return evenly spaced numbers over a specified interval. eVal='lambda'
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 #np.allclose(np.flip((eMax-eVal)), (eVal-eMin))==True
    pdf = pd.Series(pdf, index=eVal)
    return pdf


def getPCA(matrix):
    """
    Perform eigendecomposition on a Hermitian matrix.
     The eigenvalues are sorted in ascending order,
     meaning the smallest eigenvalue corresponds to the first eigenvector and
     the largest eigenvalue corresponds to the last eigenvector in the returned matrices eVal and eVec.

    Parameters:
        matrix (ndarray): Hermitian matrix.

    Returns:
        eVal (ndarray): Diagonal matrix of eigenvalues.
        eVec (ndarray): Matrix of eigenvectors.

    Example:
        eVal, eVec = getPCA(matrix)

    """
    eVal, eVec = np.linalg.eigh(matrix) #complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    eVal = np.diagflat(eVal) # identity matrix with eigenvalues as diagonal
    return eVal,eVec


def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
    '''
    Fit kernel to a series of obs, and derive the prob of obs
    x is the array of values on which the fit KDE will be evaluated
    '''
    #print(len(obs.shape) == 1)
    if len(obs.shape) == 1: obs = obs.reshape(-1,1)
    kde = KernelDensity(kernel = kernel, bandwidth = bWidth).fit(obs)
    #print(x is None)
    if x is None: x = np.unique(obs).reshape(-1,1)
    #print(len(x.shape))
    if len(x.shape) == 1: x = x.reshape(-1,1)
    logProb = kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


def visualize_pm_eig(N, T, eVal):
    """
    Visualize the eigenvalues of a correlation matrix using the Marcenko-Pastur distribution and KDE.

    Parameters:
        N (int): Number of variables or assets.
        T (int): Number of observations or time periods.
        corr (ndarray): Correlation matrix.

    Returns:
        fig: The figure object containing the plot.

    Example:
        figure = visualize_pm_eig(N, T, corr)
        figure.show()  # Display the figure
        figure.savefig('output.png')  # Save the figure to a file

    """
    Q = T / N

    pdf0 = mpPDF(1.0, Q, pts=N)
    pdf1 = fitKDE(np.diag(eVal), bWidth=0.005)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot histogram of eigenvalues
    ax.hist(np.diag(eVal), density=True, bins=150, label="Eigenvalues", alpha=0.8)

    # Plot Marcenko-Pastur distribution
    ax.plot(pdf0.keys(), pdf0, color='r', label="Marcenko-Pastur pdf")

    # Plot empirical KDE
    ax.plot(pdf1.keys(), pdf1, color='g', label="Empirical: KDE")

    # Set plot title
    ax.set_title("Eigenvalues Distribution", fontsize=16)

    # Set axis labels
    ax.set_xlabel("Eigenvalues", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    # Set y-axis scale to log
    ax.set_yscale('log')

    # Set legend
    ax.legend(loc="upper right", fontsize=10)

    # Adjust spacing
    fig.tight_layout()

    return fig


def visualize_mode(eVec, corr, mode=-1):
    """
    Visualize the mode of a correlation matrix.

    Parameters:
        eVec (ndarray): Matrix of eigenvectors.
        corr (ndarray): Correlation matrix.
        mode (int): Mode index to visualize. Default is -1, which plots the last mode.

    Returns:
        fig: The figure object containing the plot.

    Example:
        figure = visualize_mode(eVec, corr, mode=2)
        figure.show()  # Display the figure
        figure.savefig('output.png')  # Save the figure to a file

    """
    fig, ax = plt.subplots(figsize=(16, 9))

    plt.xlabel("Index")
    plt.ylabel("Intensity")

    ax.plot(eVec[:, int(mode)].flatten(), marker='x', linestyle='-')
    ax.axhline(0, c="green")
    if mode == -1:
        ax.set_ylim(-0.05, 0.2)
        ax.axhline([1 / (len(corr)) ** 0.5], c="red")

    ax.set_title("Mode Visualization", fontsize=16)

    # Adjust spacing
    fig.tight_layout()

    return fig


def correlation(returns):
    corr_matrix = returns.corr(method="pearson")
    corr_matrix = corr_matrix.replace([np.nan, np.inf, -np.inf], 0)  #clean the matrix

    return corr_matrix


if __name__ == "__main__":
    N = 1000
    T = 10000

    x = np.random.normal(0, 1, size=(T, N))
    cor = np.corrcoef(x, rowvar=0)
    eVal, eVec = getPCA(cor)

    mp_gaussian = visualize_pm_eig(N, T, cor)
    mp_gaussian.show()  # Display the figure

    mode_1 = visualize_mode(eVec, cor, mode=-1)
    mode_1.show()  # Display the figure