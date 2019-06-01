# -*- coding: utf-8 -*-

import numpy as np
from impulcifer import read_wav, write_wav, magnitude_response, plot_ir
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
import matplotlib.pyplot as plt


def main():
    smoothing = 'none'

    if False:
        fs, data = read_wav('data/Sweep-48000-16-M-5.0s-fade.wav')
        data = data[0, :]
    elif False:
        fs, data = read_wav('data/Sweep-48000-32bit-1ch-5s-20Hz-24000Hz.wav')
        data = data[0, :]
    else:
        fs = 48000
        duration = 5
        f_min = 20
        ire = ImpulseResponseEstimator(duration, low=f_min, fs=fs)
        data = ire.test_signal.copy()
        write_wav(
            'data/Sweep-{fs}-{bit}bit-1ch-{t}s-{f_min}Hz-{f_max}Hz.wav'.format(
                fs=fs,
                bit=32,
                t=duration,
                f_min=f_min,
                f_max=fs//2
            ),
            fs,
            np.expand_dims(data, 0)
        )
        seq_data_l = np.concatenate([np.zeros(fs*2), data, np.zeros(fs*9)])
        seq_data_r = np.concatenate([np.zeros(fs*9), data, np.zeros(fs*2)])
        seq_data = np.vstack([seq_data_l, seq_data_r])
        write_wav(
            'data/Sweep-sequence-{fs}-{bit}bit-2ch-{t}s-{f_min}Hz-{f_max}Hz.wav'.format(
                fs=fs,
                bit=32,
                t=duration,
                f_min=f_min,
                f_max=fs // 2
            ),
            fs,
            seq_data
        )

    f, m = magnitude_response(data, fs)
    f = f[1:]
    m = m[1:]
    fr = FrequencyResponse(name='Magnitude', frequency=f, raw=m)
    fr.plot_graph(show=True, close=False, smoothed=True, color=None, f_min=10, f_max=fs / 2)

    plot_ir(data[-int(fs*0.1):], fs, max_time=None, show_plot=True)


if __name__ == '__main__':
    main()
