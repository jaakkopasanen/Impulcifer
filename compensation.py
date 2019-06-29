# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from hrir import HRIR
from impulse_response_estimator import ImpulseResponseEstimator
from autoeq.frequency_response import FrequencyResponse


def main():
    # Read files
    estimator = ImpulseResponseEstimator.from_wav('data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav')

    # Raw HRIR measurement
    raw_hrir = HRIR(estimator)
    raw_hrir.open_recording('data/compensation2/recording.wav', speakers=['FL', 'FR'])

    # HRIR virtualized with headphones including headphone compensation
    virtualized_hrir = HRIR(estimator)
    virtualized_hrir.open_recording('data/compensation/headphones-virtualized-compensated.wav', speakers=['FL', 'FR'])

    # Headphones without virtualization
    headphones_hrir = HRIR(estimator)
    headphones_hrir.open_recording('data/compensation/headphones.wav', speakers=['FL', 'FR'])

    # Calculate frequency response of headphones
    headphones_left = headphones_hrir.irs['FL']['left']
    f, m = headphones_left.magnitude_response()
    headphones_left = FrequencyResponse(name='Left', frequency=f[1:], raw=m[1:])
    headphones_left.interpolate()
    headphones_left.center()

    headphones_right = headphones_hrir.irs['FR']['right']
    f, m = headphones_right.magnitude_response()
    headphones_right = FrequencyResponse(name='Left', frequency=f[1:], raw=m[1:])
    headphones_right.interpolate()
    headphones_right.center()

    # Figure and axes
    fig, ax = plt.subplots(2, 2)

    for i, speaker in enumerate(['FL', 'FR']):
        for j, side in enumerate(['left', 'right']):
            # Select single impulse responses
            raw = raw_hrir.irs[speaker][side]
            virtualized = virtualized_hrir.irs[speaker][side]

            # Calculate raw HRIR frequency respones
            raw_f, raw_m = raw.magnitude_response()
            raw_fr = FrequencyResponse(name='Raw', frequency=raw_f[1:], raw=raw_m[1:])
            raw_fr.interpolate()
            raw_fr.center()
            raw_fr.smoothen_fractional_octave(window_size=1/12)

            # Calculate virtualized HRIR frequency response
            virtualized_f, virtualized_m = virtualized.magnitude_response()
            virtualized_fr = FrequencyResponse(name='Raw', frequency=virtualized_f[1:], raw=virtualized_m[1:])
            virtualized_fr.interpolate()
            virtualized_fr.center()
            virtualized_fr.smoothen_fractional_octave(window_size=1/12)

            # Replace raw data with smoothed for plotting
            raw_fr.raw = raw_fr.smoothed.copy()
            raw_fr.smoothed = []
            virtualized_fr.raw = virtualized_fr.smoothed.copy()
            virtualized_fr.smoothed = []

            # Difference between frequency responses of raw HRIR and virtualized HRIR
            error = FrequencyResponse(name='Error', frequency=raw_fr.frequency, raw=virtualized_fr.raw - raw_fr.raw)

            # Plot frequency responses
            raw_fr.plot_graph(fig=fig, ax=ax[i, j], show=False, color=None)
            virtualized_fr.plot_graph(fig=fig, ax=ax[i, j], show=False, color=None)
            error.plot_graph(fig=fig, ax=ax[i, j], show=False, color='red')
            if side == 'left':
                headphones_left.plot_graph(fig=fig, ax=ax[i, j], show=False, color=None)
            if side == 'right':
                headphones_right.plot_graph(fig=fig, ax=ax[i, j], show=False, color=None)
            ax[i, j].set_title(f'{speaker}-{side}')
            ax[i, j].legend(['Raw', 'Virtualized', 'Error', 'Headphones'])

    plt.show()


if __name__ == '__main__':
    main()
