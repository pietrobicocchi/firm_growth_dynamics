from src.csv_data.extract import *
from src.csv_data.extract import *

import numpy as np
import pandas as pd
from src.export import *



def df_to_sales(df):
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df[['GVKEY', 'datadate', 'saleq', 'prccq']]
    aggregated = df.groupby([df["datadate"], df["GVKEY"]]).agg({"saleq": "sum", "prccq": "first"}).unstack()

    sales = aggregated["saleq"].resample("Q").sum()
    prices = aggregated["prccq"].resample("Q").last()

    return sales, prices


def drop_companies(df, min_length):
    df = df.drop(df.columns[df.apply(lambda col: col.notnull().sum() < min_length)], axis=1)

    return df


def clean_returns(sales, min_length=160, difference=4):
    sales = sales.replace(0, np.nan)
    sales = drop_companies(df=sales, min_length= min_length)  # Dropping companies
    sales = sales.fillna(0)

    log_ret = np.log(sales).diff(difference)
    log_ret.replace([np.inf, -np.inf], np.nan, inplace=True)

    resc_log_ret = (log_ret - log_ret.mean()) / log_ret.std()
    resc_log_ret = resc_log_ret.fillna(0)

    return resc_log_ret


if __name__ == "__main__":
    # Clean data
    df = extract_data()
    sales, prices = df_to_sales(df)
    returns_sales = clean_returns(sales)
    returns_prices = clean_returns(prices)


    # Save the cleaned data
    path_clean_s = '/Users/pietrobicocchi/Desktop/project/data/cleaned/CompStat_sales.pkl'
    path_clean_p = '/Users/pietrobicocchi/Desktop/project/data/cleaned/CompStat_prices.pkl'
    save_cleaned_data(returns_sales, path_clean_s)
    save_cleaned_data(returns_prices, path_clean_p)

    # Load the saved DataFrame
    returns_prices = load_cleaned_data(path_clean_s)

    # Now you can use the loaded DataFrame
    print(returns_prices.head())


