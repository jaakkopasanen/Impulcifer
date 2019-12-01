# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from utils import write_wav
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
TEST_SIGNAL = os.path.join(DIR_PATH, 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl')


def main():
    # Open HRIR
    estimator = ImpulseResponseEstimator.from_pickle(TEST_SIGNAL)
    hrir = HRIR(estimator)
    hrir.open_recording(os.path.join(DIR_PATH, 'FL,FR.wav'), speakers=['FL', 'FR'])
    
    # Create test signal sequence
    speakers = ['FL', 'FR']
    seq_data = estimator.sweep_sequence(speakers, 'stereo')

    # Virtualize sine sweep sequence with HRIR
    virtualized = []
    for i, speaker in enumerate(speakers):
        track = seq_data[i, :]
        virtualized.append(np.sum([
            hrir.irs[speaker]['left'].convolve(track),
            hrir.irs[speaker]['right'].convolve(track)
        ], axis=0))
    virtualized = np.vstack(virtualized)

    # Normalized to 0 dB
    virtualized /= np.max(np.abs(virtualized))

    # Write virtualized sequence to disk
    file_path = os.path.join(DIR_PATH, f'headphones-sweep-sequence-{estimator.file_name(32)}.wav')
    write_wav(file_path, estimator.fs, virtualized, bit_depth=32)


if __name__ == '__main__':
    main()
