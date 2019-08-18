# Sharpe ratio
# This calculates the main statistics of a portfolio, based on the tickers and distribution mentioned during
# course's lesson 7 and 8

# SOLVED: I have a problem with plot that overlaps what should be different graphics

import pandas as pd
import math
import matplotlib.pyplot as plt


from Lesson_05 import get_data, plot_data, compute_daily_returns
from Lesson_03 import normalize_data


def main_stats_single_asset(_asset, _k=252):
    """
    Input:
    _asset: Series with stock/portfolio data . The data in _asset must be one year
    data or less, for the values like Sharpe Ratio to be valid
    For daily samples, _k=252. For weekly samples, _k=52, for monthly samples, _k=12
    _k is NOT the number of periods sampled, it is a constant that depends only on the sampling period

    :return: Main stock/portfolio statistics:
    1. Cumulative return
    2. Average daily returns
    3. Risk (Returns standard deviation)
    4. Kurtosis
    5. Sharpe Ratio

    """

    import math

    # Calculation of cumulative returns
    # _norm_asset = normalize_data(_asset[1:])
    # _asset_value = pd.DataFrame()
    # _asset_value['Return'] = _norm_asset.sum()
    # _cumulative_returns = (_asset_value['Return'] / _asset_value.iloc[0, 0]) - 1
    _asset_return = ((_asset.iloc[-1] / _asset.iloc[0]) - 1)

    # Portfolio Daily returns
    _daily_returns = _asset.pct_change()
    # To avoid division by zero, we eliminate the first return
    _daily_returns = _daily_returns[1:]
    # Average Daily Return
    _avg_daily_return = _daily_returns.mean()

    # Standard deviation of daily returns (risk)
    _asset_risk = _daily_returns.std()

    # Kurtosis
    _asset_kurtosis = _daily_returns.kurtosis()

    # Sharpe Ratio
    # For daily samples, sqrt(252). For weekly samples, sqrt(52), for monthly samples, sqrt(12)
    _asset_sr = math.sqrt(_k) * _avg_daily_return / _asset_risk

    return _asset_return, _avg_daily_return, _asset_risk, _asset_kurtosis, _asset_sr


def main_stats_portfolio(_asset, _k=252):
    """
    Input:
    _asset: Dataframe with stock/portfolio data. The data in _asset must be one year data or less, for the values
    like Sharpe Ratio to be valid. Each column must have the amount in USD of the ticker that conform the portfolio
    For daily samples, _k=252. For weekly samples, _k=52, for monthly samples, _k=12
    _k is NOT the number of periods sampled, it is a constant that depends only on the sampling period

    :return: Main stock/portfolio statistics:
    1. Cumulative return
    2. Average daily returns
    3. Risk (Returns standard deviation)
    4. Kurtosis
    5. Sharpe Ratio

    """

    import math

    # Calculation of cumulative returns
    _norm_asset = normalize_data(_asset[1:])
    _asset_value = pd.DataFrame()
    _asset_value['Return'] = _asset.sum(axis=1)
    # _cumulative_returns = (_asset_value['Return'] / _asset_value.iloc[0, 0]) - 1
    _asset_return = (_asset_value.iloc[-1, 0] / _asset_value.iloc[0, 0]) - 1

    # Portfolio Daily returns
    _daily_returns = compute_daily_returns(_asset_value)
    # To avoid division by zero, we eliminate the first return
    _daily_returns = _daily_returns[1:]
    # Average Daily Return
    _avg_daily_return = _daily_returns.mean().values[0]

    # Standard deviation of daily returns (risk)
    _asset_risk = _daily_returns.std().values[0]

    # Kurtosis
    _asset_kurtosis = _daily_returns.kurtosis().values[0]

    # Sharpe Ratio
    # For daily samples, sqrt(252). For weekly samples, sqrt(52), for monthly samples, sqrt(12)
    _asset_sr = math.sqrt(_k) * _avg_daily_return / _asset_risk

    return _asset_return, _avg_daily_return, _asset_risk, _asset_kurtosis, _asset_sr


def test_run():
    # Compute portfolio statistics
    # Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'GOOG', 'XOM', 'GLD']
    allocs = [.4, .4, .1, .1]
    start_val = 10000.0
    components = get_data(symbols, dates)
    # plt.figure(1)
    plot_data(components, title='Portfolio Components Prices')

    # Normalize prices
    norm_prices = normalize_data(components[1:])
    # plt.figure(2)
    plot_data(norm_prices, title="Portfolio Components Normalized Prices", ylabel="Normalized Prices")

    # Allocate tickers weight in portfolio
    alloced = norm_prices * allocs
    # Position values (investment value)
    pos_vals = alloced * start_val
    # plt.figure(3)
    plot_data(pos_vals, title="Portfolio Position Values", ylabel="Position Values")
    # plt.close()

    # Portfolio value
    port_val = pd.DataFrame()
    port_val['Return'] = pos_vals.sum(axis=1)
    plt.figure()
    plot_data(port_val['Return'], title="Portfolio Value", ylabel="Portfolio Value")
    # plt.close()

    # Portfolio Daily returns
    daily_rets = compute_daily_returns(port_val)
    # To avoid division by zero, we eliminate the first return
    daily_rets = daily_rets[1:]

    # Main portfolio statistics:
    # Cumulative returns
    port_cum_rets = (port_val['Return'] / port_val.iloc[0, 0]) - 1
    plt.figure()
    plot_data(port_cum_rets, title="Portfolio Cumulative Returns", ylabel="Portfolio Cumulative Returns")
    port_return = (port_val.iloc[-1, 0] / port_val.iloc[0, 0]) - 1
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

    ar1, dr1, risk1, kurt1, sr1 = main_stats_portfolio(pos_vals)
    ar2, dr2, risk2, kurt2, sr2 = main_stats_single_asset(port_val)

    print(ar1 == ar2)
    print(dr1 == dr2)
    print(risk1 == risk2)
    print(kurt1 == kurt2)
    print(sr1 == sr2)


if __name__ == "__main__":
    test_run()
