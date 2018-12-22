# Optimizers: Building a parameterized model
# Minimize an objective function, using Scipy

#######################################################################################################
# WARNING: Run the code by segments, the plots overlap and as of 12/22/2018 I still don't know how
# to handle the plots so it is a different plot per segment
#######################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def f(X):
    """ Given a scalar X, return some value (a real number) """
    Y = (X - 1.5)**2 + 0.5
    print('X = {}, Y = {}'.format(X, Y))  # for tracing
    return Y


def error(line, data):
    """ Compute error between given line model and observed data
    Parameters
    ----------
    line: tuple/list/array (C0, C1) where C0 is slope and C1 is Y-intercept
    data: 2D array where each row is a point (x,y)

    Returns error as a single real value
    """

    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err


def fit_line(data, error_func):
    """ Fit a line to given data, using supplied error function.
     Parameters
     ----------
    data: 2D array where each row is a point (x,y)
    error_func: function that computes the error between a line and observed data

    Returns line that minimizes the error function.
    """

    # Generate initial guess for line model
    l = np.float32([0, np.mean(data[:, 1])])  # slope = 0, intercept = mean(y values)

    # Plot initial guess (optional)
    x_ends = np.float32([-5,5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label='Initial Guess')

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp':True})
    return result.x


def error_poly(C, data):
    """ Compute error between given polynomial and observed data

    Parameters
    ----------
    C: numpy.poli1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value
    """

    # Metric: sum of squared Y-axis differences
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err


def fit_poly(data, error_func, degree=3):
    """ Fit a polynomial to given data, using supplied error function.
     Parameters
     ----------
    data: 2D array where each row is a point (x,y)
    error_func: function that computes the error between a polynomial and observed data

    Returns polynomial that minimizes the error function.
    """

    # Generate initial guess for polynomial model (all coeffs = 1)
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    # Plot initial guess (optional)
    x = np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label='Initial Guess')

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp':True})
    return np.poly1d(result.x)  # Convert optimal result into a poly1d object and


def test_run():
    #######################################################################################################
    # WARNING: Run the code by segments, the plots overlap and as of 12/22/2018 I still don't know how
    # to handle the plots so it is a different plot per segment
    #######################################################################################################

    #######################################################################################################
    # Segment 1: Calculate and plot the minima of a function
    #######################################################################################################
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    print('Minima found at: ')
    print('X = {}, Y = {}'.format(min_result.x, min_result.fun))

    # Plot function values, mark minima
    Xplot = np.linspace(0.5, 2.5, 21)
    Yplot = f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title('Minima of an objective function')
    plt.show()

    #######################################################################################################
    # Segment 2: Fit line to given data points
    #######################################################################################################

    # Define original line
    l_orig = np.float32([4, 2])
    print('Original line: C0 = {}, C1 = {}'.format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b', linewidth=2.0, label='Original line')

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data Points')

    # Try to fit a line to this data
    l_fit = fit_line(data, error)
    print('Fitted line: C0 = {}, C1 = {}'.format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label='Fitted line')

    # Add legend and show plot
    plt.legend(loc='upper left')
    plt.show()

    #######################################################################################################
    # Segment 3: Fit polynomial to given data points
    #######################################################################################################

    # Define original polynomial
    C_orig = np.poly1d(np.float32([1.5, -10, -5, 60, 50]))
    # This is the polynomial printed using the default method for the poly1d class:
    print(C_orig)
    Xorig = np.linspace(-6, 6, 21)
    Yorig = C_orig(Xorig)
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label='Original line')
    plt.ylim(-500, 2000)

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data Points')

    # Try to fit a line to this data
    poly_fit = fit_poly(data, error_poly, len(C_orig))
    print('Fitted polynomial: \n', poly_fit)
    Yfitted = poly_fit(Xorig)
    plt.plot(data[:, 0], Yfitted, 'r--', linewidth=2.0, label='Fitted polynomial')

    # Add legend and show plot
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    test_run()
