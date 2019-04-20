# -*- coding: utf-8 -*-

import numpy as np
import pyfftw
import scipy.signal
from scipy.fftpack import fft
from scipy.signal import fftconvolve
from time import time


def main(n):
    a = pyfftw.empty_aligned(n, dtype='complex128')
    a[:] = np.random.random(n)
    b = pyfftw.empty_aligned(n, dtype='complex128')
    b[:] = np.random.random(n)
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack
    c = scipy.signal.fftconvolve(a, b)
    t = time()
    pyfftw.interfaces.scipy_fftpack.fft(c)
    print('pyfftw scipy fft {:.1f}ms'.format((time() - t)*1000))
    t = time()
    np.fft.fft(c)
    print('native scipy fft {:.1f}ms'.format((time() - t)*1000))


if __name__ == '__main__':
    main(5*48000)
