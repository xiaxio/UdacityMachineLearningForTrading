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
import numpy as np
import pandas as pd
from Lesson_03 import download_yahoo_historical_data, symbol_to_path, symbol_to_path_xlsx


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
    """ Computes main statistics for each stock in _tickers_list, and updates summary_file
    Stock value is expected to be in 'Adj Close' column (yahoo style) """

    from Lesson_08 import main_stats_single_asset

    _file_name = 'summary_file'
    _summary_data = pd.read_csv(symbol_to_path(_file_name, _analysis_directory), index_col='Symbol')
    for _ticker in _tickers_list:
        print('Calculating main stats for ', _ticker)
        _base_df = pd.DataFrame(index=_dates)
        _ticker_data = pd.read_csv(symbol_to_path(_ticker, _tickers_directory), index_col='Date')
        _ticker_data = _base_df.join(_ticker_data).dropna()
        _ticker_ar, _ticker_dr, _ticker_risk, _ticker_kurtosis, _ticker_sr = \
            main_stats_single_asset(_ticker_data['Adj Close'])
        _summary_data.loc[_ticker, 'Ret52w'] = _ticker_ar
        _summary_data.loc[_ticker, 'AvgDailyRet52w'] = _ticker_dr
        _summary_data.loc[_ticker, 'Risk52w'] = _ticker_risk
        _summary_data.loc[_ticker, 'Kurtosis52w'] = _ticker_kurtosis
        _summary_data.loc[_ticker, 'SharpeRatio52w'] = _ticker_sr
        _summary_data.loc[_ticker, 'MinAdjCl52w'] = _ticker_data.describe().loc['min', 'Adj Close']
        _summary_data.loc[_ticker, 'MaxAdjCl52w'] = _ticker_data.describe().loc['max', 'Adj Close']

    _file_name = 'summary_file'
    _summary_data.to_csv(symbol_to_path(_file_name, _analysis_directory))
    print('Summary file created')

    return


def insert_ta(_tickers_list, _dates, _tickers_directory, _analysis_directory):
    """ Computes main technical analysis values, and saves new tickers files with Technical Analysis (TA) """

    from functools import partial

    _file_name = 'summary_file'
    _summary_data = pd.read_csv(symbol_to_path(_file_name, _analysis_directory), index_col='Symbol')

    for _ticker in _tickers_list:
        print('Calculating technical analysis values for ', _ticker)
        _ticker_data = pd.read_csv(symbol_to_path(_ticker, _tickers_directory), index_col='Date',
                                   parse_dates=True, na_values=['nan'])
        _ticker_data['DailyRet'] = _ticker_data['Adj Close'].pct_change()
        _ticker_data.loc[_ticker_data.index[0], 'DailyRet'] = 0

        # Insert Simple Moving Average columns
        for _window_size in (5, 20, 50, 200):
            _ticker_data['SMA_' + str(_window_size)] = _ticker_data['Adj Close'].rolling(_window_size).mean()
            # Previous Periods Data (avoid back-testing bias)
            _ticker_data['SMA_' + str(_window_size) + '(-1)'] = _ticker_data['SMA_' + str(_window_size)].shift(periods=1)
            _ticker_data['SMA_' + str(_window_size) + '(-2)'] = _ticker_data['SMA_' + str(_window_size)].shift(periods=2)

        # Insert Weighted Moving Average columns
        # Formula for EWMA:
        # https://stackoverflow.com/questions/38836482/create-a-rolling-custom-ewma-on-a-pandas-dataframe

        # _alpha = 1 - np.log(2) / 3  # This is ewma's decay factor.

        for _window_size in (5, 20, 50, 200):
            # In the end, I used _alpha as suggested for EMA.
            # The recommended _alpha = 1 - np.log(2) / 3 is too close to price action
            # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
            _alpha = 2 / (_window_size + 1)
            _weights = list(reversed([(1 - _alpha) ** n for n in range(_window_size)]))
            ewma = partial(np.average, weights=_weights)
            _ticker_data['EWMA_' + str(_window_size)] = _ticker_data['Adj Close'].rolling(_window_size).apply(ewma, raw=True)
            # Previous Periods Data (avoid back-testing bias)
            _ticker_data['EWMA_' + str(_window_size) + '(-1)'] = _ticker_data['EWMA_' + str(_window_size)].shift(periods=1)
            _ticker_data['EWMA_' + str(_window_size) + '(-2)'] = _ticker_data['EWMA_' + str(_window_size)].shift(periods=2)

        # Save ticker data with Technical Analysis
        _ticker_data.to_csv(symbol_to_path(_ticker + '_TA', _tickers_directory))

    return


def simulate_strategy_on_ticker(_ticker, _ticker_data, _column_1, _column_2, _analysis_directory, _strategy_number):
    """ Receives two columns names and compares their values, adding a column with Trading Signals, and another
     with Trading Strategy """

    # Generate Trading Signals (buy=1 , sell=-1, do nothing=0)
    _signal_column = 'Strategy_' + str(_strategy_number) + '_signal'
    _ticker_data[_signal_column] = 0
    _signal_column_number = _ticker_data.columns.get_loc(_signal_column)
    _signal = 0

    for i, r in enumerate(_ticker_data.iterrows()):
        if r[1][_column_1 + '(-2)'] < r[1][_column_2 + '(-2)'] and r[1][_column_1 + '(-1)'] > r[1][_column_2 + '(-1)']:
            _signal = 1
        elif r[1][_column_1 + '(-2)'] > r[1][_column_2 + '(-2)'] and r[1][_column_1 + '(-1)'] < r[1][_column_2 + '(-1)']:
            _signal = -1
        else:
            _signal = 0
        _ticker_data.iloc[i, _signal_column_number] = _signal

    # Generate Trading Strategy (own stock=1 , not own stock=0, short-selling not available yet)
    _strategy_column = 'Strategy_' + str(_strategy_number) + '_flag'
    _ticker_data[_strategy_column] = 1  # By default, assumes the strategy condition is originally met
    _strategy_column_number = _ticker_data.columns.get_loc(_strategy_column)
    _strategy_flag = 0

    for i, r in enumerate(_ticker_data.iterrows()):
        if r[1][_signal_column] == 1:
            _strategy_flag = 1
        elif r[1][_signal_column] == -1:
            _strategy_flag = 0
        else:
            _strategy_flag = _ticker_data[_strategy_column][i - 1]
        _ticker_data.iloc[i, _strategy_column_number] = _strategy_flag

    # Strategy daily returns without Trading Commissions
    _strategy_daily_returns_column = 'Strategy_' + str(_strategy_number) + '_stratDayRet'
    _ticker_data[_strategy_daily_returns_column] = _ticker_data['DailyRet'] * _ticker_data[_strategy_column]

    # Strategy cumulative returns without Trading Commissions
    _strategy_cum_returns_column = 'Strategy_' + str(_strategy_number) + '_stratCumRet'
    _ticker_data[_strategy_cum_returns_column] = np.cumprod(_ticker_data[_strategy_daily_returns_column] + 1) - 1

    # Calculate strategy's last year main statistics
    _one_year_ago = (pd.datetime.today() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    _end_date = pd.datetime.today().strftime('%Y-%m-%d')
    _one_year = pd.date_range(_one_year_ago, _end_date)
    _base_df = pd.DataFrame(index=_one_year)
    _tmp_ticker_data = _base_df.join(_ticker_data[_strategy_daily_returns_column]).dropna()

    # Strategy cumulative returns without Trading Commissions 52 weeks
    _strategy_cum_returns_52w_column = 'Strategy_' + str(_strategy_number) + '_stratCumRet_52w'
    _ticker_data[_strategy_cum_returns_52w_column] = np.cumprod(_tmp_ticker_data[_strategy_daily_returns_column] + 1) - 1
    _strategy_cum_return_52w = _ticker_data[_strategy_cum_returns_52w_column][-1]

    # Strategy average daily returns
    _strategy_daily_avg_return = _tmp_ticker_data[_strategy_daily_returns_column].mean()

    # Strategy risk (standard deviation) without Trading Commissions 52 weeks
    _strategy_risk = _tmp_ticker_data[_strategy_daily_returns_column].std()

    # Strategy Sharpe Ratio
    _strategy_sharpe_ratio = np.sqrt(252) * _strategy_daily_avg_return / _strategy_risk

    # Save strategy's main statistics in summary file
    _file_name = 'summary_file'
    _summary_data = pd.read_csv(symbol_to_path(_file_name, _analysis_directory), index_col='Symbol')

    # Whole data stats
    _summary_data.loc[_ticker, _strategy_column + '_AvgDailyRet'] = _ticker_data[_strategy_daily_returns_column].mean()
    _summary_data.loc[_ticker, _strategy_column + '_CumRet'] = _ticker_data[_strategy_cum_returns_column][-1]
    _summary_data.loc[_ticker, _strategy_column + '_risk'] = _ticker_data[_strategy_daily_returns_column].std()

    # Last year stats
    _summary_data.loc[_ticker, _strategy_column + '_AvgDailyRet_52w'] = _strategy_daily_avg_return
    _summary_data.loc[_ticker, _strategy_column + '_CumRet_52w'] = _strategy_cum_return_52w
    _summary_data.loc[_ticker, _strategy_column + '_risk_52w'] = _strategy_risk
    _summary_data.loc[_ticker, _strategy_column + '_SharpeRatio_52w'] = _strategy_sharpe_ratio
    _summary_data.to_csv(symbol_to_path(_file_name, _analysis_directory))

    return _ticker_data


def run_strategies(_tickers_list, _tickers_directory, _analysis_directory):
    """ Runs strategies over _tickers_list, and computes returns and main statistics for the backtest simulation """

    _MAs = ['SMA_', 'EWMA_']
    _fast_moving_avg = [5, 5, 5, 20, 20, 50]
    _slow_moving_avg = [20, 50, 200, 50, 200, 200]
    _strategy = pd.DataFrame(columns=['Number', 'Entry', 'Exit', 'Filter'])
    _st_counter = 0

    for _ma in _MAs:

        for _counter in range(len(_fast_moving_avg)):
            # Strategy n: _fast_moving_avg and _slow_moving_avg crossover
            _strategy.loc[_st_counter, 'Number'] = _st_counter + 1
            _strategy.loc[_st_counter, 'Entry'] = _ma + str(_fast_moving_avg[_counter]) + ' > ' + _ma + str(_slow_moving_avg[_counter])
            _strategy.loc[_st_counter, 'Exit'] = _ma + str(_fast_moving_avg[_counter]) + ' < ' + _ma + str(_slow_moving_avg[_counter])
            _strategy.loc[_st_counter, 'Filter'] = 'None'
            print('Strategy info:')
            print(_strategy.loc[_st_counter, :])

            column_1 = _ma + str(_fast_moving_avg[_counter])
            column_2 = _ma + str(_slow_moving_avg[_counter])
            for _ticker in _tickers_list:
                print('Simulating strategy with ticker ' + _ticker)
                _ticker_data = pd.read_csv(symbol_to_path(_ticker + '_TA', _tickers_directory), index_col='Date',
                                           parse_dates=True, na_values=['nan'])
                _ticker_data = simulate_strategy_on_ticker(_ticker, _ticker_data, column_1, column_2,
                                                           _analysis_directory, _st_counter)
                _ticker_data.to_csv(symbol_to_path(_ticker + '_TA', _tickers_directory))
            _st_counter += 1

    _strategy.to_csv(symbol_to_path('strategies_info', _analysis_directory), index=False)

    return


def summarize_the_summary(_analysis_directory):
    """ Reads the summary file, generates statistics about them, and saves them in another file """

    # Read summary file
    # _file_name = 'summary_file_' + pd.datetime.today().strftime('%Y%m%d')
    _file_name = 'summary_file'
    # _file_name = 'C:\\users\\aroom\\documents\\Data\\summaries\\summary_file_20190817'
    _summary_data = pd.read_csv(symbol_to_path(_file_name, _analysis_directory), index_col='Symbol')

    # Extracts column names to create new DataFrame with only the columns with the string '_flag_'
    _columns_names = pd.DataFrame(_summary_data.columns, columns=['Column_Name'])
    _filtered_columns_names = _columns_names[(_columns_names['Column_Name'].str.contains('_flag_'))]
    _sim_output = _summary_data.loc[:, _filtered_columns_names['Column_Name'].values]

    # Creates summary DataFrame with main statistics
    _sim_output_describe = _sim_output.describe()
    # _sim_output_describe.loc['mean']
    # _sim_output_describe.transpose()

    # Extracts index names to create new DataFrame with only the rows with the string '_AvgDailyRet'
    _index_names = pd.DataFrame(_sim_output_describe.transpose().index, columns=['Index_Name'])
    _filtered_index_names = _index_names[(_index_names['Index_Name'].str.contains('_AvgDailyRet'))]
    _AvgDailyRet = _sim_output_describe.transpose().loc[_filtered_index_names['Index_Name'].values, :]
    # _AvgDailyRet.to_csv(symbol_to_path('Strategies_AvgDailyRet', _analysis_directory))

    # Extracts index names to create new DataFrame with only the rows with the string '_SharpeRatio'
    _filtered_index_names = _index_names[(_index_names['Index_Name'].str.contains('_SharpeRatio'))]
    _SharpeRatio = _sim_output_describe.transpose().loc[_filtered_index_names['Index_Name'].values, :]
    # _SharpeRatio.to_csv(symbol_to_path('Strategies_SharpeRatio', _analysis_directory))

    # Extracts index names to create new DataFrame with only the rows with the string '_risk'
    _filtered_index_names = _index_names[(_index_names['Index_Name'].str.contains('_risk'))]
    _risk = _sim_output_describe.transpose().loc[_filtered_index_names['Index_Name'].values, :]
    # _risk.to_csv(symbol_to_path('Strategies_Risk', _analysis_directory))

    # Extracts index names to create new DataFrame with only the rows with the string '_CumRet'
    _filtered_index_names = _index_names[(_index_names['Index_Name'].str.contains('_CumRet'))]
    _CumRet = _sim_output_describe.transpose().loc[_filtered_index_names['Index_Name'].values, :]
    # _CumRet.to_csv(symbol_to_path('Strategies_CumRet', _analysis_directory))

    # Create excel file for the summary describe file
    with pd.ExcelWriter(symbol_to_path_xlsx('summary_describe_' + pd.datetime.today().strftime('%Y%m%d'),
                                            _analysis_directory)) as writer:
        _AvgDailyRet.to_excel(writer, sheet_name='AvgDailyRet')
        _CumRet.to_excel(writer, sheet_name='CumRet')
        _risk.to_excel(writer, sheet_name='Risk')
        _SharpeRatio.to_excel(writer, sheet_name='SharpeRatio')

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
    # file_name = 'summary_file_' + pd.datetime.today().strftime('%Y%m%d')
    file_name = 'summary_file'
    filtered_tickers_data.to_csv(symbol_to_path(file_name, config.analysis_directory))

    download_tickers_historical_data(filtered_tickers_list, dates, config.tickers_directory)

    one_year_ago = (pd.datetime.today() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    one_year = pd.date_range(one_year_ago, end_date)
    calculate_main_stats(filtered_tickers_list, one_year, config.tickers_directory, config.analysis_directory)
    insert_ta(filtered_tickers_list, one_year, config.tickers_directory, config.analysis_directory)
    run_strategies(filtered_tickers_list, config.tickers_directory, config.analysis_directory)
    summarize_the_summary(config.analysis_directory)


"""
    _tickers_list = filtered_tickers_list
    _date = one_year
    _date = dates
    _tickers_directory = config.tickers_directory
    _analysis_directory = config.analysis_directory
    _ticker = _tickers_list[1]
    _window_size = 5
    _counter = 0
    _column_1 = column_1
    _column_2 = column_2
"""

if __name__ == "__main__":
    main()


