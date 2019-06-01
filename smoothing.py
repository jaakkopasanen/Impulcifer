# -*- coding: utf-8 -*-

from impulcifer import read_wav, magnitude_response
from autoeq.frequency_response import FrequencyResponse
import matplotlib.pyplot as plt


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
    fr.plot_graph(show=True, close=False, smoothed=False, color=None)
    plt.plot(fr.frequency, fr.smoothed, linewidth=2)
    plt.show()


if __name__ == '__main__':
    main()
