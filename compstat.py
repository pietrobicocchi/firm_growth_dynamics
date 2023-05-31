import src.csv_data.extract as ex
from src.csv_data.utils import *

plt.rcParams["figure.figsize"] = [10,10]  # Set default figure size


if __name__ == '__main__':
    df = ex.extract_data()
    #dict_dfs = clean(df)
    #print(dict_dfs)

