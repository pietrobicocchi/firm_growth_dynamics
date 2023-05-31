import pandas as pd
from pathlib import Path

# read environment variables from the .env file instead of from your local environment.
import os
from dotenv import load_dotenv
#env_path = Path('.', '.env')
# Get the current working directory
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parpar_dir = os.path.dirname(parent_dir)

# Construct the file path to the .env file in the main project folder
env_path = os.path.join(parpar_dir, '.env')
load_dotenv(dotenv_path=env_path) # take environment variables from .env.
DATA_DIR = os.getenv('DATA_DIR')


def get_project_root() -> Path:
    """
    Returns always project root directory.
    """
    return Path(__file__).parent.parent.parent


def extract_data(data_path: str = None):
    """This function extracts the desired information from the data file given.

    Args:
        data_path (str, optional): Path where one can read experimental measurements. Defaults to None.
    Returns:
        Dataframe.
    """
    if data_path is None:
        # by default, data are stored in the data folder
        data_path = str(get_project_root()) + str(DATA_DIR) + "/CompStat/crsp_ccm_inventories_sales.csv"

    # read the data
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    print(get_project_root())