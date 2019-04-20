# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from impulcifer import magnitude_response


def main():
    Fs, data = wavfile.read('data/HE400S.wav')
    data = np.array(data)
    print(Fs)
    print(data.shape)
    print(data.dtype)

    t = np.arange(0, len(data) / Fs, 1 / Fs)
    dtype = data.dtype
    data = data.astype('float32')
    if dtype == 'int32':
        data /= 2.0**31
    if dtype == 'int16':
        data /= 2.0**15
    elif dtype == 'uint8':
        data /= 2.0**8
        data *= 2.0
        data -= 1.0

    ax = plt.subplot(2, 2, 1)
    plt.plot(t, data[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    ax = plt.subplot(2, 2, 3)
    plt.plot(t, data[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    ax = plt.subplot(2, 2, (2, 4))
    f, X_mag_l = magnitude_response(data[:, 0], Fs)
    f, X_mag_r = magnitude_response(data[:, 1], Fs)
    X_mag_l = X_mag_l[f <= 20000]
    X_mag_r = X_mag_r[f <= 20000]
    f = f[f <= 20000]
    X_mag_l -= X_mag_l[np.argmin(np.abs(f - 1000.0))]
    X_mag_r -= X_mag_r[np.argmin(np.abs(f - 1000.0))]
    plt.plot(f, X_mag_l, f, X_mag_r)
    plt.semilogx()
    plt.xlim([20, 20000])
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dBr)')
    plt.legend(['Left', 'Right'])
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    plt.show()


if __name__ == '__main__':
    main()
