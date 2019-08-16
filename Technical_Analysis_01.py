######################################################
# Stock Technical Analysis with Python               #
#                                                    #
# File management module for symbols                 #
# Reads a file with a list of tickers and creates    #
# an individual file for each ticker with data       #
# downloaded from yahoo.com                          #
#                                                    #
# Aroom Gonzalez                                     #
#                                                    #
# Current version: 0.0                               #
#                                                    #
######################################################

import datetime
import pandas as pd
from Lesson_03 import download_yahoo_historical_data, symbol_to_path


def clean_tickers_list(_tickers_data):
    """ Removes tickers that contain dots or other symbols, which yahoo does not support """
    # https://stackoverflow.com/questions/34055584/python-pandas-string-contains-and-doesnt-contain
    # df[(df.str.contains("b") == True) & (df.str.contains("a") == False)]
    # For some reason, I have to use '\.' instead of '.'. If I use '.', it takes all the values as True

    _clean_tickers_data = _tickers_data[(_tickers_data['Symbol'].str.contains('\.') == False)]

    return _clean_tickers_data


def append_detailed_info(_tickers_data):
    """ Includes additional info for each stock, downloaded from yahoo.com API """
    # https://stackoverflow.com/questions/54815864/downloading-a-companies-market-cap-from-yahoo
    from pandas_datareader import data

    _tickers_list = _tickers_data['Symbol'].values
    _tickers_extended_data = _tickers_data.set_index('Symbol')
    print('Downloading Market Cap data')
    _tmp_df = data.get_quote_yahoo(_tickers_list)['marketCap']
    _tickers_extended_data = _tickers_extended_data.join(_tmp_df)
    print('Downloading Dividend Date data')
    _tmp_df = data.get_quote_yahoo(_tickers_list)['dividendDate']
    _tickers_extended_data = _tickers_extended_data.join(pd.to_datetime(_tmp_df, unit='s'))
    print('Downloading Annual Dividend Yield data')
    _tmp_df = data.get_quote_yahoo(_tickers_list)['trailingAnnualDividendYield']
    _tickers_extended_data = _tickers_extended_data.join(_tmp_df)
    print('Downloading Annual Dividend Rate data')
    _tmp_df = data.get_quote_yahoo(_tickers_list)['trailingAnnualDividendRate']
    _tickers_extended_data = _tickers_extended_data.join(_tmp_df)
    print('Downloading Average Daily Volume data')
    _tmp_df = data.get_quote_yahoo(_tickers_list)['averageDailyVolume3Month']
    _tickers_extended_data = _tickers_extended_data.join(_tmp_df)

    return _tickers_extended_data


def get_tickers_list(_file_name):
    _tickers_list = pd.DataFrame()
    _tickers_list = pd.read_csv(_file_name)

    return _tickers_list


def filter_tickers_list(_tickers_data):
    """ Filters tickers to work only with the ones that meet the criteria, and so reduce processing time """
    import config
    _filtered_tickers = _tickers_data[(_tickers_data['marketCap'] > config.minimum_market_cap)]
    _filtered_tickers = _tickers_data[(_tickers_data['averageDailyVolume3Month'] > config.minimum_average_volume)]
    # _filtered_tickers = _tickers_data[(_tickers_data['averageDailyVolume3Month'] > 1000000)]

    return _filtered_tickers


def download_tickers_historical_data(_tickers_list, _dates, _tickers_directory):
    """ Download a list of tickers, and saves data in specified directory """
    from matplotlib import pyplot as plt
    _counter = 1
    for _ticker in _tickers_list:
        print('Downloading data for ', _ticker)
        df_temp = download_yahoo_historical_data(_ticker, _dates[0], _dates[-1])
        df_temp = df_temp.dropna()
        df_temp.to_csv(symbol_to_path(_ticker, _tickers_directory))
        if (_counter % 20 == 0) and (_counter > 1):
            print('Waiting 20 seconds...')
            plt.pause(20)
            print('Wait time ended')


def calculate_main_stats(_tickers_list, _dates, _tickers_directory, _analysis_directory):
    """ Computes main statistics for each stock in _tickers_list, and updates summary_file"""

    from Lesson_08 import main_stats

    _file_name = 'summary_file_' + pd.datetime.today().strftime('%Y%m%d')
    _summary_data = pd.read_csv(symbol_to_path(_file_name, _analysis_directory), index_col='Symbol')
    for _ticker in _tickers_list:
        print('Calculating main stats for ', _ticker)
        _base_df = pd.DataFrame(index=_dates)
        _ticker_data = pd.read_csv(symbol_to_path(_ticker, _tickers_directory), index_col='Date')
        _ticker_data = _base_df.join(_ticker_data).dropna()
        _ticker_ar, _ticker_dr, _ticker_risk, _ticker_kurtosis, _ticker_sr = main_stats(_ticker_data)
        _summary_data.loc[_ticker, 'Ret52w'] = _ticker_ar
        _summary_data.loc[_ticker, 'AvgDailyRet52w'] = _ticker_dr
        _summary_data.loc[_ticker, 'Risk52w'] = _ticker_risk
        _summary_data.loc[_ticker, 'Kurtosis52w'] = _ticker_kurtosis
        _summary_data.loc[_ticker, 'SharpeRatio52w'] = _ticker_sr
        _summary_data.loc[_ticker, 'MinAdjCl52w'] = _ticker_data.describe().loc['min', 'Adj Close']
        _summary_data.loc[_ticker, 'MaxAdjCl52w'] = _ticker_data.describe().loc['max', 'Adj Close']

    _file_name = 'summary_file_' + pd.datetime.today().strftime('%Y%m%d')
    _summary_data.to_csv(symbol_to_path(_file_name, _analysis_directory))
    print('Summary file created')

    return


def insert_ta(_tickers_list, _dates, _tickers_directory, _analysis_directory):
    """ Computes main technical analysis values, and updates summary_file"""

    from Lesson_05 import compute_daily_returns

    _file_name = 'summary_file_' + pd.datetime.today().strftime('%Y%m%d')
    _summary_data = pd.read_csv(symbol_to_path(_file_name, _analysis_directory), index_col='Symbol')

    for _ticker in _tickers_list:
        print('Calculating technical analysis values for ', _ticker)
        _df_tmp = pd.read_csv(symbol_to_path(_ticker, _tickers_directory), index_col='Date',
                              parse_dates=True, na_values=['nan'])
        _df_tmp['DailyRet'] = _df_tmp['Adj Close'].pct_change()
        _df_tmp.loc[_df_tmp.index[0], 'DailyRet'] = 0
        for _window_size in (5, 20, 50, 200):
            _df_tmp['SMA_' + str(_window_size)] = _df_tmp['Adj Close'].rolling(_window_size).mean()

    return


def main():
    # Load tickers list as specified in config.py
    import config

    tickers_data = pd.DataFrame()
    tickers_data = get_tickers_list(config.tickers_file)
    tickers_data = clean_tickers_list(tickers_data)
    tickers_list = tickers_data['Symbol'].values
    tickers_data = append_detailed_info(tickers_data)

    end_date = pd.datetime.today().strftime('%Y-%m-%d')
    start_date = (pd.datetime.today() - pd.Timedelta(days=365*4)).strftime('%Y-%m-%d')  # 4 years back
    dates = pd.date_range(start_date, end_date)

    filtered_tickers_data = filter_tickers_list(tickers_data)
    filtered_tickers_list = filtered_tickers_data.index.values
    file_name = 'summary_file_' + pd.datetime.today().strftime('%Y%m%d')
    filtered_tickers_data.to_csv(symbol_to_path(file_name, config.analysis_directory))

    download_tickers_historical_data(filtered_tickers_list, dates, config.tickers_directory)

    one_year_ago = (pd.datetime.today() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    one_year = pd.date_range(one_year_ago, end_date)
    calculate_main_stats(filtered_tickers_list, one_year, config.tickers_directory, config.analysis_directory)
    insert_ta(filtered_tickers_list, config.tickers_directory, config.analysis_directory)


if __name__ == "__main__":
    main()



