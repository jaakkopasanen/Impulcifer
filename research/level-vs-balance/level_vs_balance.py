# -*- coding: utf-8 -*-

import os
import sys
from glob import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from autoeq.frequency_response import FrequencyResponse
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from utils import save_fig_as_png, config_fr_axis

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


def main():
    test_signal = os.path.join(DIR_PATH, 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl')
    estimator = ImpulseResponseEstimator.from_pickle(test_signal)

    for group in ['volume2', 'volume2-48-52', 'objective2', 'None']:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 9)
        config_fr_axis(ax)
        ax.set_title(group)

        files = sorted(
            list(glob(os.path.join(DIR_PATH, group, 'headphones*.wav'))),
            key=lambda x: float(re.search(r'\d+', os.path.split(x)[1])[0])
        )
        for file_path in files:
            hp = HRIR(estimator)
            hp.open_recording(file_path, ['FL', 'FR'])
            left = hp.irs['FL']['left'].frequency_response()
            right = hp.irs['FR']['right'].frequency_response()
            ax.plot(
                left.frequency,
                right.raw - left.raw,
                label=os.path.split(file_path)[1].replace('.wav', ''),
                linewidth=0.5
            )
        ax.legend()
        ax.set_ylim([-5, 5])
        plt.show()
        save_fig_as_png(os.path.join(DIR_PATH, f'{group}.png'), fig)


if __name__ == '__main__':
    main()
