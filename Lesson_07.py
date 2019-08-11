# Histograms and scatter plots
# To save environment (requirements) of the project in a file:
# https://www.jetbrains.com/help/pycharm/managing-dependencies.html
""" Plot a histogram """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Lesson_05 import get_data, plot_data


def compute_daily_returns(df):
    """ Compute and returns the daily return values """
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    # Original code, but uses ix. I will replace with iloc to see if it works
    # daily_returns.ix[0, :] = 0  # set daily returns for row 0 to 0
    daily_returns.iloc[0, :] = 0  # set daily returns for row 0 to 0
    return daily_returns


def test_run():
    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns", ylabel="Daily Returns")

    # Plot a histogram
    daily_returns.hist(bins=20)

    # Get mean and standard deviation
    mean = daily_returns['SPY'].mean()
    print('mean =', mean)
    std = daily_returns['SPY'].std()
    print('std =', std)

    # plots a vertical line where x=mean()
    plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    # compute kurtosis:
    # Positive value means we have fat tails => we have more values far from mean than gaussian distribution
    # negative means in general the data is closer to the mean than gaussian (normal) distribution
    print(daily_returns.kurtosis())

    #####################################################################################################
    # Plot two histograms
    #####################################################################################################
    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns", ylabel="Daily Returns")

    # Plot a histogram
    daily_returns['SPY'].hist(bins=20, label='SPY')
    daily_returns['XOM'].hist(bins=20, label='XOM')
    plt.legend(loc='upper right')

    #####################################################################################################
    # Scatter plots
    #####################################################################################################
    # Beta means how reactive is a stock compared to the market (>1 more volatile, <1 less volatile)
    # Beta is the value of the slope in a linear regression of a scatter plot from a stock returns vs SP500 returns
    # Alpha > 0 => Stock returns on average are greater than SP500
    # Alpha < 0 => Stock performing a little bit less than the market overall
    # Correlation IS NOT THE SLOPE of the linear regression
    # Correlation is how tightly the dots in the scatter plot fit the linear regression line
    # Correlation = 0: Not correlated at all
    # Correlation = 1: Very highly correlated

    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns", ylabel="Daily Returns")

    # Scatter plot SPY vs XOM
    daily_returns.plot(kind="scatter", x='SPY', y='XOM')
    # np.polyfit returns two parameters:
    # The first, is the polynomial coefficient (slope in case of grade 1). y = mx + b. This would be m
    # The second is the intersect. This would be b in y = mx + b
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1) # 1 os for function of grade 1
    plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, '-', color = 'r')
    plt.show()

    # Scatter plot SPY vs GLD
    daily_returns.plot(kind="scatter", x='SPY', y='GLD')
    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1) # 1 os for function of grade 1
    plt.plot(daily_returns['SPY'], beta_GLD * daily_returns['SPY'] + alpha_GLD, '-', color = 'r')
    plt.show()

    # Calculate correlation coefficient
    # pearson is the most commonly used method to calculate correlation
    print(daily_returns.corr(method='pearson'))


if __name__ == "__main__":
    test_run()
