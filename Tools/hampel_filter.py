import numpy as np
from numba import jit


@jit(nopython=True)
def _for_hampel_jit(array_copy, windowsize, n):
    k = 1.4826

    for idx in range(windowsize // 2, len(array_copy) - windowsize // 2):
        window = array_copy[idx - windowsize // 2:idx + windowsize // 2 + 1]
        median = np.median(window)  # calculate window median
        sigma = k * np.median(np.abs(window - median))  # calculate window Median Absolute Deviation (MAD)
        if np.abs(array_copy[idx] - median) > n * sigma:  # if the value at "idx" index is an outlier...
            array_copy[idx] = median  # replace it with the median of the window


def hampel_jit(array_copy, windowsize, n=3):
    array_copy = np.pad(array_copy, windowsize//2, mode='edge')

    _for_hampel_jit(array_copy, windowsize, n)
    return array_copy[windowsize//2:-windowsize//2]


if __name__ == '__main__':
    from time import time
    for i in range(10):
        start = time()
        hampel_jit(np.arange(1000000), 10, 3)
        print("Jit:", time() - start)
