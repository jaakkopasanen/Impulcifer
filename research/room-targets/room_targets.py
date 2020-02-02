# -*- coding: utf-8 -*-

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from autoeq.frequency_response import FrequencyResponse
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from utils import optimize_png_size

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


def main():
    # Open files
    loudspeaker = FrequencyResponse.read_from_csv(os.path.join(DIR_PATH, 'harman-in-room-loudspeaker-target.csv'))
    headphone = FrequencyResponse.read_from_csv(os.path.join(DIR_PATH, 'harman-in-room-headphone-target.csv'))
    headphone.raw += headphone.create_target(bass_boost_gain=2.2, bass_boost_fc=105, bass_boost_q=0.76)

    fig, ax = loudspeaker.plot_graph(show=False)
    headphone.plot_graph(fig=fig, ax=ax, color='blue')
    plt.show()

    for fr in [loudspeaker, headphone]:
    #for fr in [loudspeaker]:
        fr.interpolate(f_step=1.01, f_min=10, f_max=20000)
        fr.center()
        smooth = fr.copy()
        smooth.smoothen_fractional_octave(window_size=1, treble_window_size=1)
        smooth.raw = smooth.smoothed.copy()
        smooth.smoothed = np.array([])
        smooth.write_to_csv(os.path.join(DIR_PATH, os.pardir, os.pardir, 'data', f'{fr.name}.csv'))
        smooth.plot_graph()
        fr.raw[fr.frequency < 300] = fr.raw[np.argmin(np.abs(fr.frequency - 300))]
        fr.smoothen_fractional_octave(window_size=1, treble_window_size=1)
        fr.raw = fr.smoothed.copy()
        fr.smoothed = np.array([])
        #fr.raw += fr.create_target(bass_boost_gain=6.8, bass_boost_fc=105, bass_boost_q=0.76)
        fr.write_to_csv(os.path.join(DIR_PATH, os.pardir, os.pardir, 'data', f'{fr.name}-wo-bass.csv'))
        fig, ax = fr.plot_graph(show=False)
        #smooth.plot_graph(fig=fig, ax=ax, show=False, color='blue')
        #ax.legend(['Shelf', 'Original'])
        plt.show()


if __name__ == '__main__':
    main()
