import datetime
import pandas_datareader.data as web

def get_stock_data_from_web(stock_name, start_year, start_month, start_day, end_year, end_month, end_day):
    start = datetime.datetime(start_year, start_month, start_day)
    end = datetime.datetime(end_year, end_month, end_day)
    df = web.DataReader(stock_name, "yahoo", start, end)
    return df


print(get_stock_data_from_web("XOM", 2019, 10,1,2020,1,1))