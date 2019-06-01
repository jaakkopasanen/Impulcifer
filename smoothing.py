# -*- coding: utf-8 -*-

from impulcifer import read_wav, magnitude_response
from autoeq.frequency_response import FrequencyResponse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    fs, data = read_wav('data/demo/hrir.wav')
    data = data[0, :]
    f, m = magnitude_response(data, fs)
    f = f[1:]
    m = m[1:]
    fr = FrequencyResponse(name='Magnitude', frequency=f, raw=m)
    fr.interpolate()
    smoothing = 'variable'
    if smoothing == 'psychoacoustic':
        # Psychoacoustic smoothing, good for headphone compensation
        fr.smoothen(
            window_size=1 / 3,
            iterations=1,
            treble_window_size=1 / 6,
            treble_iterations=1,
            treble_f_lower=100,
            treble_f_upper=1000
        )
    elif smoothing == 'variable':
        # Variable smoothing, good for room correction
        fr.smoothen(
            window_size=1 / 9,
            iterations=1,
            treble_window_size=1 / 3,
            treble_iterations=1,
            treble_f_lower=100,
            treble_f_upper=10000
        )
    fig, ax = plt.subplots()
    plt.plot(fr.frequency, fr.raw, linewidth=0.5)
    plt.plot(fr.frequency, fr.smoothed, linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.semilogx()
    plt.xlim([20, 20000])
    plt.ylabel('Amplitude (dBr)')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    plt.show()


if __name__ == '__main__':
    main()
