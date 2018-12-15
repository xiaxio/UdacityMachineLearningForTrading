# Statistical analysis of time series
# https://classroom.udacity.com/courses/ud501/lessons/4156938722/concepts/45439393860923

"""
Global statistics:
    mean: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mean.html
    median: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.median.html
    std: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.std.html
    sum: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sum.html
    [more]: http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats

Rolling statistics:
    rolling_mean: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.rolling_mean.html (Deprecated)
    rolling_std: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.rolling_std.html (Deprecated)
    [more]:http://pandas.pydata.org/pandas-docs/stable/computation.html?highlight=rolling%20statistics#moving-rolling-statistics-moments
    Updated function example: https://stackoverflow.com/questions/50313698/pandas-rolling-mean-not-working

Document on how to import functions from other files in python (and nice explanation on objects OOP):
    https://www.csee.umbc.edu/courses/331/fall10/notes/python/python3.ppt.pdf

    Pycon 2010 webpage: http://us.pycon.org/2010/conference/schedule/event/50/
"""
"""Bollinger Bands."""

import os
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window, center=False).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # DONE: Compute and return rolling standard deviation
    return values.rolling(window=window, center=False).std()


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # DONE: Compute upper_band and lower_band
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # http://us.pycon.org/2010/conference/schedule/event/50/
    # DONE: Your code here
    # Note: Returned DataFrame must have the same number of rows
    df_tmp = df.pct_change()
    df_tmp.iloc[0, :] = 0
    return df_tmp


def compute_cumulative_returns(df):
    """Compute and return the daily return values."""
    # http://us.pycon.org/2010/conference/schedule/event/50/
    # DONE: Your code here
    # Note: Returned DataFrame must have the same number of rows
    # Solution taken from https://stackoverflow.com/questions/40811246/pandas-cumulative-return-function
    # df.ix["Cumulative"] = ((df.fillna(0) + 1).cumprod() - 1).iloc[-1] # This gives only one value per column, is not a column of Cumulative returns
    # df_tmp.iloc[0, :] = 0
    # Solution taken from https://stackoverflow.com/questions/40204396/plot-cumulative-returns-of-a-pandas-dataframe
    df_tmp = ((df + 1).cumprod() - 1)
    return df_tmp


def test_run():
    # Read data
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)

    #######################################################################
    #  Compute Bollinger Bands
    #######################################################################
    # 1. Compute rolling mean
    rm_SPY = get_rolling_mean(df['SPY'], window=20)

    # 2. Compute rolling standard deviation
    rstd_SPY = get_rolling_std(df['SPY'], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
    rm_SPY.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    #######################################################################
    # Compute daily returns
    #######################################################################
    # Read data
    dates = pd.date_range('2012-07-01', '2012-07-31')  # one month only
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

    #######################################################################
    # Compute cumulative returns
    #######################################################################
    # Read data
    dates = pd.date_range('2012-01-01', '2012-12-31')  # whole 2012
    symbols = ['SPY']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)

    # Compute cumulative returns
    cum_returns = compute_cumulative_returns(daily_returns)
    plot_data(cum_returns, title="Cumulative returns", ylabel="Cumulative returns")


if __name__ == "__main__":
    test_run()
