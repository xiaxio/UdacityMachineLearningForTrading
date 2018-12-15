import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web


def download_yahoo_historical_data(dyhd_symbol, start_date, end_date):
    """
    Downloads historical data using yahoo API

    :param dyhd_symbol: symbol name as str
    :param start_date:
    :param end_date:
    :return: historical data supplied by yahoo's API
    :rtype: dataframe
    """
    while True:
        try:
            dyhd_symbol_data = web.DataReader(dyhd_symbol, 'yahoo', start_date, end_date)
            break
        except IndexError:
            print('Ticker ' + dyhd_symbol + ' not found by yahoo. IndexError')
            dyhd_symbol_data = []
            return dyhd_symbol_data

    return dyhd_symbol_data


def normalize_data(df):
    """ Normalize stock prices using the first row of the dataframe """
    return df / df.ix[0, :]


def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    df_temp = df.ix[start_index:end_index, columns]
    plot_data(df_temp, "Selected Data")
    return


def symbol_to_path(symbol, base_dir="C:\\Users\\aroom\\Documents\\Data\\tickers_data\\"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def save_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_temp = pd.DataFrame(index=dates)

    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = download_yahoo_historical_data(symbol, dates[0], dates[-1])
        df_temp = df_temp.dropna()
        df_temp.to_csv(symbol_to_path(symbol))

    return df_temp


def test_run():
    # Initialize variables
    df = pd.DataFrame()

    # Define a date range
    start_date = '2009-01-01'
    end_date = '2014-12-31'
    dates = pd.date_range(start_date, end_date)

    # Choose stock symbols to read
    symbols = ['SPY', 'GOOG', 'IBM', 'GLD', 'XOM']

    # Download data and save data
    save_data(symbols, dates)

    # Get stock data
    df = get_data(symbols, dates)

    # Slice and plot
    plot_selected(df, ['SPY', 'IBM'], '2010-03-01', '2010-04-01')

    df1 = normalize_data(df)
    plot_data(df1, "Normalized Data")


if __name__ == "__main__":
    test_run()