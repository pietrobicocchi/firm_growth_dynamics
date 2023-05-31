import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


def import_data(tickers: List[str], start: str, end: str, interval: str) -> Dict[str, pd.DataFrame]:
    """
    Imports financial data from Yahoo Finance for the given tickers within the specified time period.

    Args:
        tickers (List[str]): List of ticker symbols of the financial instruments.
        start (str): Start date of the data in 'YYYY-MM-DD' format.
        end (str): End date of the data in 'YYYY-MM-DD' format.
        interval (str): Time interval for the data, such as '1d' for daily, '1wk' for weekly, etc.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where the keys are tickers and the values are pandas DataFrames
            containing the imported financial data.

    Raises:
        ValueError: If the start or end dates are not in the correct format.
        Exception: If an error occurs while retrieving the data from Yahoo Finance.

    Example:
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        start_date = '2022-01-01'
        end_date = '2022-12-31'
        interval = '1d'

        data = import_data(tickers, start_date, end_date, interval)
    """
    interv = str(interval)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # empty dictionary to store the DataFrames
    dfs = {}

    # collect into a DataFrame
    for ticker in tickers:
        df = pdr.get_data_yahoo(f'{ticker}', pd.to_datetime(start), pd.to_datetime(end), interval=interv, repair=True)
        dfs[ticker] = df

    return dfs


def listed_data(dfs, tickers, interval):
    """
    Filters and returns a dictionary of DataFrames containing listed financial data based on the specified interval.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary where the keys are tickers and the values are pandas DataFrames
            containing the financial data.
        tickers (List[str]): List of ticker symbols of the financial instruments.
        interval (str): Time interval for the data, such as 'day', 'week', 'month', or 'quart'.

    Returns:
        Dict[str, pd.DataFrame]: A filtered dictionary where the keys are tickers and the values are pandas DataFrames
            containing the listed financial data within the specified interval.

    Raises:
        None

    Example:
        dfs = {
            'AAPL': df_aapl,
            'GOOGL': df_googl,
            'MSFT': df_msft
        }
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        interval = 'day'

        listed_dfs = listed_data(dfs, tickers, interval)
    """

    global start, end

    def to_delete(dfs, start, end, interval):
        """
        Identifies tickers and their start dates to be deleted based on the specified interval.
        """
        to_del = {}
        if interval == 'day':
            for ticker, df in dfs.items():
                start_date = df.index[0]
                end_date = df.index[-1]
                # print(f"{ticker} start date: {start_date}")
                if start_date > pd.to_datetime(start) + timedelta(days=5) or end_date < pd.to_datetime(end) - timedelta(
                        days=5):  # safeness timedelta
                    to_del[ticker] = start_date
        if interval == 'week':
            for ticker, df in dfs.items():
                start_date = df.index[0]
                end_date = df.index[-1]
                # print(f"{ticker} start date: {start_date}")
                if start_date > pd.to_datetime(start) + timedelta(days=15) or end_date < pd.to_datetime(
                        end) - timedelta(days=15):  # safeness timedelta
                    to_del[ticker] = start_date
        if interval == 'month':
            for ticker, df in dfs.items():
                start_date = df.index[0]
                end_date = df.index[-1]
                # print(f"{ticker} start date: {start_date}")
                if start_date > pd.to_datetime(start) + timedelta(days=34) or end_date < pd.to_datetime(
                        end) - timedelta(days=34):  # safeness timedelta
                    to_del[ticker] = start_date
        if interval == 'quart':
            for ticker, df in dfs.items():
                start_date = df.index[0]
                end_date = df.index[-1]
                # print(f"{ticker} start date: {start_date}")
                if start_date > pd.to_datetime(start) + timedelta(days=95) or end_date < pd.to_datetime(
                        end) - timedelta(days=95):  # safeness timedelta
                    to_del[ticker] = start_date
        return to_del

    def listed_tickers(tickers, to_del):
        listed_true = [x for x in tickers if x not in to_del.keys()]
        return listed_true

    def delete_unlisted(dfs, to_del):
        for key in to_del.keys():
            dfs.pop(key)
        return dfs

    to_del = to_delete(dfs, start, end, interval)
    listed_true = listed_tickers(tkrs, to_del)
    dfs = delete_unlisted(dfs, to_del)

    return dfs


def extract_tickers(df: pd.DataFrame) -> List[str]:
    """
    Extracts the ticker symbols from a DataFrame and returns them as a list.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing financial data.

    Returns:
        List[str]: A list of ticker symbols extracted from the DataFrame.

    Raises:
        None

    Example:
        data = {
            'AAPL': [100, 101, 102],
            'GOOGL': [200, 201, 202],
            'MSFT': [300, 301, 302]
        }
        df = pd.DataFrame(data)

        tickers = extract_tickers(df)
    """
    tickers = list(df.keys())

    return tickers


def ind_ticks_to_sect(tickers: List[str]) -> List[str]:
    """
    Retrieves the sector information for a list of ticker symbols and returns a list of corresponding sectors.
    """

    tick_sect = []

    for tik in tickers:
        a = yf.Ticker(str(tik))
        tick_sect.append(a.info['sector'])
        # print(f'{tik} is okay')
    return tick_sect


def sectors(tk_ind: List[str]) -> Tuple[np.ndarray, Dict[str, List[int]], List[int], List[str]]:
    """
    Creates sector indices and labels based on a list of ticker sectors.

    Args:
        tk_ind (List[str]): A list of sectors assigned to each ticker.

    Returns:
        Tuple[np.ndarray, Dict[str, List[int]], List[int], List[str]]: A tuple containing the following:
            - idx_by_sector (np.ndarray): An array of indices representing the sectors sorted in ascending order.
            - sector_dict (Dict[str, List[int]]): A dictionary mapping each sector to a list of corresponding indices.
            - sector_indices (List[int]): A list of sector indices corresponding to the ticker sectors.
            - sector_labels (List[str]): A list of sector labels.

    Raises:
        None

    Example:
        tk_ind = ['Technology', 'Financial', 'Technology', 'Healthcare', 'Financial']

        idx_by_sector, sector_dict, sector_indices, sector_labels = sectors(tk_ind)
    """

    idx_by_sector = np.argsort(tk_ind)  # tick is my list that assign at each ticker the corresponding sector

    # create dictionary mapping sectors to tickers
    sector_dict = {}
    for i in range(len(tk_ind)):
        sector = tk_ind[i]
        if sector not in sector_dict:
            sector_dict[sector] = []
        sector_dict[sector].append(i)

    # sort sector names alphabetically
    sector_names = sorted(sector_dict.keys())

    # create list of sector labels and corresponding indices
    sector_labels = []
    sector_indices = []
    for sector in sector_names:
        sector_labels.append(sector)
        for i in idx_by_sector:
            if i in sector_dict[sector]:
                sector_indices.append(idx_by_sector.tolist().index(i) + 1)  # +1 just to translate it graphically
                break
    return idx_by_sector, sector_dict, sector_indices, sector_labels