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
from utils import read_wav, optimize_png_size, sync_axes

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

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18, 9)

    # Set axis
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_title(f'{["Left", "Right"][j]} ear - {["left", "right"][j-i]} transducer')
            ax.set_xlabel('Frequency (Hz)')
            ax.semilogx()
            ax.set_xlim([20, 20e3])
            ax.set_ylabel('Amplitude (dB)')
            ax.grid(True, which='major')
            ax.grid(True, which='minor')
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # Plot
    for name, ir_pair in responses.items():
        row = 1 if 'reversed' in name else 0
        for col, side in enumerate(['left', 'right']):
            fr = ir_pair[side].frequency_response()
            axs[row, col].plot(fr.frequency, fr.raw, linewidth=0.5)

    # Sync axes limits
    sync_axes([ax for row in axs for ax in row])

    # Save figure
    file_path = os.path.join(DIR_PATH, 'Results.png')
    fig.savefig(file_path, bbox_inches='tight')
    optimize_png_size(file_path)

    # Show figure
    plt.show()


if __name__ == '__main__':
    main()
