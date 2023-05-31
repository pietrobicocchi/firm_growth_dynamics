import yfinance as yf
from src.export import *
from src.yahoo.extract import *
from src.yahoo.utils import *
from src.random_matrix import *

yf.pdr_override()  # it is used to mantain all the functionalities of Yahoo data reader library

plt.rcParams["figure.figsize"] = [10, 10]  # Set default figure size

from pathlib import Path
import os
from dotenv import load_dotenv
env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path) # take environment variables from .env.
DATA_DIR = os.getenv('DATA_DIR')
US_DIR = os.getenv("USER_PATH")

if __name__ == "__main__":
    # load returns
    path_clean = path_clean = str(str(US_DIR)+str(DATA_DIR)+"/cleaned/cleaned_data.pkl")
    returns_daily = load_cleaned_data(path_clean)

    # what is the industry of each stock
    tickers = extract_tickers(returns_daily)
    tkrs_industry = ind_ticks_to_sect(tickers)
    idx_by_sector, sector_dict, sector_indices, sector_labels = sectors(tkrs_industry)

    # correlations
    corr_d = correlation(returns_daily)

    # eigendecomposition
    eVal_d, eVec_d = getPCA(corr_d)
