# Sharpe ratio
# This calculates the main statistics of a portfolio, based on the tickers and distribution mentioned during
# course's lesson 7 and 8

import pandas as pd
import math
import matplotlib.pyplot as plt


from Lesson_05 import get_data, plot_data, compute_daily_returns
from Lesson_03 import normalize_data


def test_run():
    # Compute portfolio statistics
    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'GOOG', 'XOM', 'GLD']
    allocs = [.4, .4, .1, .1]
    start_val = 10000.0
    components = get_data(symbols, dates)
    plot_data(components, title='Portfolio Components Prices')

    # Normalize prices
    norm_prices = normalize_data(components[1:])
    plot_data(norm_prices, title="Portfolio Components Normalized Prices", ylabel="Normalized Prices")

    # Allocate tickers weight in portfolio
    alloced = norm_prices * allocs
    # Position values (investment value)
    pos_vals = alloced * start_val
    plot_data(pos_vals, title="Portfolio Position Values", ylabel="Position Values")
    plt.close()

    # Portfolio value
    port_val = pd.DataFrame()
    port_val['Return'] = pos_vals.sum(axis=1)
    plot_data(port_val['Return'], title="Portfolio Value", ylabel="Portfolio Value")

    # Portfolio Daily returns
    daily_rets = compute_daily_returns(port_val)
    # To avoid division by zero, we eliminate the first return
    daily_rets = daily_rets[1:]

    # Main portfolio statistics:
    # Cumulative returns
    port_cum_rets = (port_val['Return'] / port_val.iloc[0, 0]) - 1
    plot_data(port_cum_rets, title="Portfolio Cumulative Returns", ylabel="Portfolio Cumulative Returns")
    port_return = (port_val.iloc[-1,0] / port_val.iloc[0,0]) - 1
    print('Portfolio Return: ', port_return)
    # Average Daily Return
    avg_daily_return = daily_rets.mean().values[0]
    print('Portfolio Average Daily Return:', avg_daily_return)
    # Standard deviation of daily returns (risk)
    port_risk = daily_rets.std().values[0]
    print('Portfolio Risk (std(returns)):', port_risk)
    # Sharpe Ratio
    # For daily samples, sqrt(252). For weekly samples, sqrt(52), for monthly samples, sqrt(12)
    port_sr = math.sqrt(252) * avg_daily_return / port_risk
    print('Portfolio Sharpe Ratio:', port_sr)


if __name__ == "__main__":
    test_run()