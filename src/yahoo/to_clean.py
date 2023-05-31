"""
This script clean the data and store them in a proper folder in /data using the library pickle. It mainly refers
to the functions from extract.py inside the very same folder.
"""

from src.yahoo.extract import *
from src.yahoo.utils import *
from src.export import *


if __name__ == "__main__":
    tkrs = ['AAPL', 'ABBV', 'ABT', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AIG', 'AMAT', 'AMD',
        'AMGN', 'AMT', 'AMZN', 'AON', 'APA', 'APD', 'ATVI', 'AVGO', 'AXP', 'BA',
        'BAC', 'BAX', 'BBY', 'BDX', 'BK', 'BKNG', 'BLK', 'BMY', 'C',
        'CHTR', 'CI', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMI', 'CMS',
        'COF', 'COP', 'COST', 'CSCO', 'CVS', 'CVX', 'DD', 'DE', 'DHR', 'DIS', 'DOW',
        'DTE', 'DUK', 'DVA', 'EA', 'EBAY', 'ECL', 'ED', 'EMN', 'EMR', 'EOG', 'EQR',
        'ES', 'ETN', 'EXC', 'EXPD', 'F', 'FDX', 'FE', 'FIS', 'FISV',
        'FLR', 'FMC', 'FOX', 'FOXA', 'GD', 'GE', 'GILD', 'GIS', 'GLW', 'GM', 'GOOG',
        'GOOGL', 'GPC', 'GPN', 'GS', 'GWW', 'HAL', 'HAS', 'HD', 'HES', 'HIG', 'HON',
        'HPQ', 'HUM', 'IBM', 'ICE', 'IDXX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU',
        'IP', 'IPG', 'IQV', 'ISRG', 'IT', 'ITW', 'IVZ', 'JCI', 'JNJ', 'JPM', 'K',
        'KEY', 'KEYS', 'KHC', 'KMI', 'KO', 'KR', 'LEG', 'LH', 'LIN', 'LLY',
        'LMT', 'LOW', 'LRCX', 'LUV', 'LYB', 'MA', 'MAR', 'MAS', 'MCD', 'MCHP',
        'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MMM', 'MO', 'MOS', 'MRK', 'MS', 'MSFT',
        'MU', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOV', 'NOW', 'NSC',
        'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA', 'OKE', 'OMC', 'ORCL', 'OXY',
        'PAYX', 'PCAR', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM',
        'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU',
        'PSA']


    # Interval of time
    start='2005-01-01'
    end='2022-12-23'

    # Import the data
    df_day = import_data(tkrs, start, end, interval = "1d")

    # Clean the data
    returns_d = clean_data(df_day, prd=1)

    # Save the cleaned data
    path_clean = '/Users/pietrobicocchi/Desktop/project/data/cleaned/cleaned_data.pkl'
    save_cleaned_data(returns_d, path_clean)

    # Load the saved DataFrame
    returns_clean_d = load_cleaned_data(path_clean)

    # Now you can use the loaded DataFrame
    print(returns_clean_d.head())