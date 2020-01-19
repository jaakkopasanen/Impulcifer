# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from utils import read_wav, write_wav
from constants import HEXADECAGONAL_TRACK_ORDER, HESUVI_TRACK_ORDER, SPEAKER_NAMES

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to HRIR or HeSuVi file.')
    parser.add_argument('--track_order', type=str, required=True,
                        help='Track order in HRIR file. "hesuvi" or "hexadecagonal"')
    parser.add_argument('--times', type=str, default=argparse.SUPPRESS,
                        help='Reverberation times for different channels in milliseconds. During this time the '
                             'reverberation tail will be reduced by 100 dB. A comma separated list of channel name and '
                             'reverberation time pairs, separated by colon. If only a single numeric value is given, '
                             'it is used for all channels. When some channel names are give but not all, the missing '
                             'channels are not affected. Must be at least 3 ms smaller than the HRIR length. '
                             'For example "--time=300" or '
                             '"--time=FL:500,FC:100,FR:500,SR:700,BR:700,BL:700,SL:700" or '
                             '"--time=FC:100".')
    args = parser.parse_args()
    file_path = args.file
    track_order = args.track_order
    times = dict()
    try:
        # Single float value
        times = {ch: float(args.times) / 1000 for ch in SPEAKER_NAMES}
    except ValueError:
        # Channels separated
        for ch_t in args.times.split(','):
            times[ch_t.split(':')[0].upper()] = float(ch_t.split(':')[1]) / 1000

    fs, data = read_wav(file_path)

    for ch, t in times.items():
        print(f'{ch}: {t*1000:.0f}ms')
        n_ones = int(fs * 0.003)
        n_win = int(fs * t)
        win = np.concatenate([
            np.ones(n_ones),
            signal.windows.hann(n_win * 2)[n_win:],
            np.zeros(data.shape[1] - n_ones - n_win)
        ]) - 1.0
        win *= 100  # 100 dB
        win = 10**(win / 20)  # Linear scale
        if track_order == 'hesuvi':
            tracks = [i for i in range(len(HESUVI_TRACK_ORDER)) if ch in HESUVI_TRACK_ORDER[i]]
        elif track_order == 'hexadecagonal':
            tracks = [i for i in range(len(HEXADECAGONAL_TRACK_ORDER)) if ch in HEXADECAGONAL_TRACK_ORDER[i]]
        else:
            raise ValueError(f'Invalid track_order "{track_order}", allowed values are "hesuvi" and "hexadecagonal"')
        for i in tracks:
            data[i, :] *= win

    # Write WAV
    write_wav(os.path.join(DIR_PATH, 'cropped.wav'), fs, data)


if __name__ == '__main__':
    main()
