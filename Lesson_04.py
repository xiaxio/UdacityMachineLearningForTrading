import numpy as np
import time


def get_max_index(a):
    """Return the index of the maximum value in given 1D array."""
    # https://docs.scipy.org/doc/numpy/reference/routines.sort.html
    return a.argmax()


def how_long(func, *args):
    """" Execute function with given arguments, and measures execution time """
    t0 = time()
    result = func(*args)
    t1 = time()
    return result, t1 - t0


def test_run():
    ####################################################################################
    # Arrays
    ####################################################################################
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    # List to 1D array
    print(np.array([2, 3, 4]))

    # List of tuples to 2D arrays
    print(np.array([(2, 3, 4), (5, 6, 7)]))

    # Empty arrays (they take the value that were present on the corresponding memory location)
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html
    print(np.empty(5))
    print(np.empty((5, 4)))

    # Array of 1s
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html
    print(np.ones((5, 4)))

    # Specifying the data type
    print(np.ones((5, 4), dtype=np.int_))

    # Array of 0s
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    print(np.zeros((5, 4, 3)))  # Three dimensions array

    ####################################################################################
    # Generating random numbers
    ####################################################################################
    # Generate an array full of random numbers, uniformly sampled from [0.0, 1.0)
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random.html
    print(np.random.random((5, 4)))  # pass in a size tuple
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html
    print(np.random.rand(5, 4))  # function arguments (not a tuple)

    # Sample numbers from a Gaussian (normal) distribution
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
    print(np.random.normal(size=(2,3)))  # "standard" normal (mean=0, sd=1)
    print(np.random.normal(50, 10, size=(2,3)))  # change mean to 50, and sd to 10

    # Random integers
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html
    print(np.random.randint(10))  # a single integer in [0, 10)
    print(np.random.randint(0, 10))  # Same as above, specifying [low, high) explicitly
    print(np.random.randint(0, 10, size=5))  # 5 random integers as a 1D array
    print(np.random.randint(0, 10, size=(2, 3)))  # 2x3 array of random integers

    ####################################################################################
    # Array attributes
    ####################################################################################
    a = np.random.random((5, 4))  # 5x4 array of random numbers
    print(a)
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
    print(a.shape)  # shape of the array
    print(a.shape[0])  # number of rows
    print(a.shape[1])  # number of columns
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html
    print(len(a.shape))  # This will give us the dimensions of the array
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.size.html
    print(a.size)  # number of elements in the array
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.dtype.html
    print(a.dtype)

    ####################################################################################
    # Operations on arrays
    ####################################################################################

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html
    np.random.seed(693)  # seed the random number generator
    a = np.random.randint(0, 10, size=(5, 4))  # 5x4 random integers in [0,10)
    print('Array\n', a)

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    print('Sum of all elements: ', a.sum())

    # Iterate over rows, to compute sum of each column
    print('Sum of each column:\n', a.sum(axis=0))
    # Iterate over columns, to compute sum of each row
    print('Sum of each row:\n', a.sum(axis=1))

    # Statistics: min, max, mean (across rows, cols, and overall)
    print('Minimum of each column:\n', a.min(axis=0))
    print('Maximum of each row:\n', a.max(axis=1))
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    print('Mean of all elements:\n', a.mean())

    ####################################################################################
    # Using time function
    ####################################################################################
    # https://docs.python.org/2/library/time.html#time.time
    # https://docs.python.org/2/library/timeit.html
    # https://docs.python.org/2/library/profile.html
    t1 = time.time()
    print('ML4T')
    t2 = time.time()
    print('The time taken by print statement is: ', t2 - t1, ' seconds')

    ####################################################################################
    # Slicing arrays
    ####################################################################################
    # http://docs.scipy.org/doc/numpy/reference/routines.sort.html
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # http://docs.scipy.org/doc/numpy/user/basics.creation.html
    a = np.random.rand(5, 4)
    print('Array:\n', a)

    # Accessing element at position (3,2)
    element = a[3,2]
    print(element)

    # elements in defined range
    print(a[0, 1:3])

    # top left corner
    print(a[0:2, 0:2])

    # Note: Slice n:m:t specifies a range that starts at position n, stops before m, in steps of size t
    print(a[:, 0:3:2])  # will select columns 0 and 2, for every row

    ####################################################################################
    # Accessing array elements
    ####################################################################################
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # Assigning a value to a particular location
    a[0, 0] = 1
    print('\nModified (replaced one element):\n', a)

    # Assigning a list to a column in an array
    a[:, 3] = [1, 2, 3, 4, 5]
    print('\nModified (replaced a column with a list):\n', a )

    # Accesing using list of indices
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing
    a = np.random.rand(5)
    indices = np.array([1, 1, 2, 3])
    print(a, '\n',  a[indices])

    ####################################################################################
    # Boolean or mask index arrays
    ####################################################################################
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#boolean-array-indexing

    a = np.array([(20, 25, 10, 23, 26, 32, 10, 5, 0), (0, 2, 50, 20, 0, 1, 28, 5, 0)])
    print(a)

    # We want to get all the values from the array that are less than the mean
    # Calculating mean
    mean = a.mean()
    print(mean)

    # masking
    print(a[a < mean])

    a[a < mean] = mean
    print(a)

    ####################################################################################
    # Aritmetic operations
    ####################################################################################

    # http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
    # http://docs.scipy.org/doc/numpy/user/basics.types.html
    # http://docs.scipy.org/doc/numpy/user/basics.creation.html
    # http://docs.scipy.org/doc/numpy/reference/routines.array-creation.html
    # http://docs.scipy.org/doc/numpy/user/basics.indexing.html
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    # http://docs.scipy.org/doc/numpy/reference/routines.random.html
    # http://docs.scipy.org/doc/numpy/reference/routines.math.html
    # http://docs.scipy.org/doc/numpy/reference/routines.linalg.html


if __name__ == "__main__":
    test_run()
