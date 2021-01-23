# -*- coding: utf-8 -*-

import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from utils import read_wav, config_fr_axis, optimize_png_size
from constants import COLORS
from impulse_response import ImpulseResponse

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
TEST_SIGNAL = os.path.join(DIR_PATH, 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl')


def main():
    fs, pre = read_wav(os.path.join(DIR_PATH, 'responses.wav'))
    fs, post = read_wav(os.path.join(DIR_PATH, 'hrir.wav'))
    stages = {
        'Pre': pre,
        'Post': post
    }
    room_path = os.path.join(DIR_PATH, 'room-responses.wav')
    if os.path.isfile(room_path):
        fs, room = read_wav(room_path)
        stages['Room'] = room

    for name, irs in stages.items():
        frs = []
        for i in range(4):
            fr = ImpulseResponse(irs[i], fs).frequency_response()
            fr.smoothen_fractional_octave(window_size=1/3, treble_window_size=1/3)
            frs.append(fr)

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(18, 9)
        fig.suptitle(name)

        ax[0].plot(frs[0].frequency, frs[0].smoothed, color=COLORS['blue'])
        ax[0].plot(frs[3].frequency, frs[3].smoothed, color=COLORS['red'])
        ax[0].plot(frs[0].frequency, frs[0].smoothed - frs[3].smoothed, color=COLORS['purple'])
        config_fr_axis(ax[0])
        ax[0].legend(['FL-left', 'FR-right', 'Difference'])
        ax[0].set_title('FL-left vs FR-right')

        ax[1].plot(frs[1].frequency, frs[1].smoothed, color=COLORS['blue'])
        ax[1].plot(frs[2].frequency, frs[2].smoothed, color=COLORS['red'])
        ax[1].plot(frs[1].frequency, frs[1].smoothed - frs[2].smoothed, color=COLORS['purple'])
        config_fr_axis(ax[1])
        ax[1].legend(['FL-right', 'FR-left', 'Difference'])
        ax[1].set_title('FL-right vs FR-left')

        fig_path = os.path.join(DIR_PATH, f'{name}.png')
        fig.savefig(fig_path, bbox_inches='tight')
        optimize_png_size(fig_path)

    plt.show()


if __name__ == '__main__':
    main()
