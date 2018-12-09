import numpy as np


def test_run():
    # List to 1D array
    print(np.array([2, 3, 4]))

    # List of tuples to 2D arrays
    print(np.array([(2, 3, 4), (5, 6, 7)]))

    # Empty arrays (they take the value that were present on the corresponding memory location)
    print(np.empty(5))
    print(np.empty((5, 4)))

    # Array of 1s
    print(np.ones((5,4)))

    # Specifying the data type
    print(np.ones((5, 4), dtype=np.int_))

    # Array of 0s
    print(np.zeros((5,4)))

    # Generating random numbers
    # Generate an array full of random numbers, uniformly sampled from [0.0, 1.0)
    print(np.random.random((5, 4)))  # pass in a size tuple
    print(np.random.rand(5, 4))  # function arguments (not a tuple)

    # Sample numbers from a Gaussian (normal) distribution
    print(np.random.normal(size=(2,3)))  # "standard" normal (mean=0, sd=1)
    print(np.random.normal(50, 10, size=(2,3)))  # change mean to 50, and sd to 10

    # Random integers
    print(np.random.randint(10))  # a single integer in [0, 10)
    print(np.random.randint(0, 10))  # Same as above, specifying [low, high) explicitly
    print(np.random.randint(0, 10, size=5))  # 5 random integers as a 1D array
    print(np.random.randint(0, 10, size=(2, 3)))  # 2x3 array of random integers

    # Array attributes
    a = np.random.random((5, 4))  # 5x4 array of random numbers
    print(a)
    print(a.shape)  # shape of the array
    print(a.shape[0])  # number of rows
    print(a.shape[1])  # number of columns
    print(len(a.shape))  # This will give us the dimensions of the array
    print(a.size)  # number of elements in the array
    print(a.dtype)


if __name__ == "__main__":
    test_run()
