# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import convolve, hanning
from impulcifer import read_wav, write_wav, magnitude_response, plot_ir
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
import matplotlib.pyplot as plt


def main():
    filt = hanning(20) * 0.1
    signal = np.concatenate([
        np.zeros(20),
        np.ones(50),
        np.ones(50) * -1,
        #np.zeros(50)
    ]) * 0.5
    result = convolve(signal, filt, mode='full')
    plt.plot(signal)
    plt.plot(filt)
    plt.plot(result, '-')
    plt.plot(result[:len(signal)+len(filt)], '.')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
