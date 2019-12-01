# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from utils import write_wav
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
TEST_SIGNAL = os.path.join(DIR_PATH, 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl')


def plot_stereo_track(data, fs):
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(16, 10)
    start = 0.0
    end = (data.shape[1] - 1) / fs
    t = np.arange(start, end, 1 / fs)
    for i in range(2):
        ax[i].plot(t, data[i], linewidth=0.5)
        ax[i].set_xlabel('Time (s)')
        ax[i].set_xlim([start, end])
        ax[i].set_ylim([-1, 1])
        ax[i].grid(True, which='major')
        ax[i].grid(True, which='minor')
        ax[i].set_title(['Left', 'Right'][i])
    return fig, ax


def main():
    # Open HRIR
    estimator = ImpulseResponseEstimator.from_pickle(TEST_SIGNAL)
    hrir = HRIR(estimator)
    hrir.open_recording(os.path.join(DIR_PATH, 'FL,FR.wav'), speakers=['FL', 'FR'])
    hrir.crop_heads()
    hrir.crop_tails()
    
    # Create test signal sequence
    speakers = ['FL', 'FR']
    seq_data = estimator.sweep_sequence(speakers, 'stereo')

    fig, ax = plot_stereo_track(seq_data, estimator.fs)
    fig.suptitle('Sweep sequence')

    left = np.vstack([
        hrir.irs['FL']['left'].convolve(seq_data[0]),
        hrir.irs['FL']['right'].convolve(seq_data[0])
    ])
    right = np.vstack([
        hrir.irs['FR']['left'].convolve(seq_data[1]),
        hrir.irs['FR']['right'].convolve(seq_data[1])
    ])
    virtualized = left + right

    fig, ax = plot_stereo_track(virtualized, estimator.fs)
    fig.suptitle('Sweep sequence convolved with HRIR')
    plt.show()

    # Virtualize sine sweep sequence with HRIR
    # virtualized = []
    # for i, speaker in enumerate(speakers):
    #     track = seq_data[i, :]
    #     virtualized.append(np.sum([
    #         hrir.irs[speaker]['left'].convolve(track),
    #         hrir.irs[speaker]['right'].convolve(track)
    #     ], axis=0))

    virtualized = np.vstack(virtualized)

    # Normalized to 0 dB
    virtualized /= np.max(np.abs(virtualized))

    # Write virtualized sequence to disk
    file_path = os.path.join(DIR_PATH, f'headphones-sweep-seq-{",".join(speakers)}-stereo-{estimator.file_name(32)}.wav')
    write_wav(file_path, estimator.fs, virtualized, bit_depth=32)


if __name__ == '__main__':
    main()
