import numpy as np
import pandas as pd
from typing import List, Dict

import yfinance as yf
yf.pdr_override()


def from_df_to_np(x):
    '''
    Convert DataFrame to numpy array
    '''
    return x.to_numpy().astype(float)  # astype needed to show results


def clean_data(dfs: Dict[str, pd.DataFrame], prd: int = 1) -> pd.DataFrame:
    """
    Cleans and preprocesses financial data by filling missing values, calculating returns and log-returns,
    and rescaling the log returns.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary where the keys are tickers and the values are pandas DataFrames
            containing the financial data.
        prd (int, optional): The period used to calculate returns and log-returns. Default is 1.

    Returns:
        pd.DataFrame: A new DataFrame containing the rescaled log-returns for each ticker.

    Raises:
        None

    Example:
        dfs = {
            'AAPL': df_aapl,
            'GOOGL': df_googl,
            'MSFT': df_msft
        }
        prd = 1

        cleaned_data = clean_data(dfs, prd)
    """
    # Fill NaN values with the mean value for each variable across all the tickers
    for ticker, df in dfs.items():
        means = df.mean()  # Calculate the mean value for each variable in the DataFrame
        df.fillna(value=means, inplace=True)  # Fill NaN values with the mean value

    # Returns and log-returns and rescaled log
    for ticker, df in dfs.items():
        # df['returns'] = df['Adj Close'].diff()
        df['returns'] = df['Adj Close'].pct_change(periods=prd)
        df['log_returns'] = np.log(df['Adj Close']).diff(periods=prd)
        df['resc_log_ret'] = (df['log_returns'] - df['log_returns'].mean()) / df['log_returns'].std()

    # drop the first rows - they are nan
    for ticker, df in dfs.items():
        df.drop(index=df.index[0:int(prd)], axis=0, inplace=True)

    # extract the rescaled log-returns price column from each dataframe and store them in a new dictionary
    returns = {}
    for symbol, df in dfs.items():
        df = df.replace([np.nan, np.inf, -np.inf], 0)
        returns[symbol] = df['resc_log_ret']

    # create a new dataframe containing the adjusted close price data for all stocks
    returns_df = pd.DataFrame(returns)
    #returns_df.replace([np.nan, np.inf, -np.inf], 0)

    return returns_df
