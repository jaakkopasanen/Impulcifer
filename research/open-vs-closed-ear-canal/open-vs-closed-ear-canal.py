# -*- coding: utf-8 -*-

import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from impulcifer import headphone_compensation
from impulse_response_estimator import ImpulseResponseEstimator
from impulse_response import ImpulseResponse
from utils import read_wav, optimize_png_size

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
TEST_SIGNAL = os.path.join(DIR_PATH, 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl')


def main():
    # Read all headphone responses
    responses = dict()
    for file_path in glob(os.path.join(DIR_PATH, '*', 'headphone-responses.wav')):
        name = os.path.split(os.path.split(file_path)[0])[1]
        responses[name] = dict()
        fs, data = read_wav(file_path)
        responses[name]['left'] = ImpulseResponse(data[0], fs)
        responses[name]['right'] = ImpulseResponse(data[3], fs)

    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(20, 15)
    for i, location in enumerate(['rear', 'center', 'front']):
        for j, side in enumerate(['left', 'right']):
            # Frequency responses
            cl = responses[f'closed-{location}'][side].frequency_response()
            cl.smoothen_fractional_octave(window_size=1 / 3, treble_f_lower=20000, treble_f_upper=23999)
            op = responses[f'open-{location}'][side].frequency_response()
            op.smoothen_fractional_octave(window_size=1 / 3, treble_f_lower=20000, treble_f_upper=23999)
            pl = responses[f'plugged-{location}'][side].frequency_response()
            pl.smoothen_fractional_octave(window_size=1 / 3, treble_f_lower=20000, treble_f_upper=23999)

            # Plot
            ax = axs[i, j]
            # ax.set_xlabel('Frequency (Hz)')
            ax.semilogx()
            ax.set_xlim([20, 20e3])
            ax.set_ylabel('Amplitude (dB)')
            ax.set_title(f'{location} {side}')
            ax.grid(True, which='major')
            ax.grid(True, which='minor')
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

            ax.plot(cl.frequency, cl.smoothed, color='#1f77b4')
            ax.plot(op.frequency, op.smoothed, '--', color='#1f77b4')
            ax.plot(pl.frequency, pl.smoothed, '-.', color='#1f77b4')
            ax.legend([f'Closed {location} {side}', f'Open {location} {side}', f'Plugged {location} {side}'])

    file_path = os.path.join(DIR_PATH, 'Results.png')
    fig.savefig(file_path, bbox_inches='tight')
    optimize_png_size(file_path)
    plt.show()


if __name__ == '__main__':
    main()
