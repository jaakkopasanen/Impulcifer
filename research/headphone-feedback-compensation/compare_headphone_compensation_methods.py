# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from autoeq.frequency_response import FrequencyResponse
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from utils import sync_axes, save_fig_as_png, config_fr_axis
from constants import COLORS

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
TEST_SIGNAL = os.path.join(DIR_PATH, 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl')


def main():
    estimator = ImpulseResponseEstimator.from_pickle(TEST_SIGNAL)

    # Open feedback measurement
    feedback = HRIR(estimator)
    feedback.open_recording(os.path.join(DIR_PATH, 'headphones-FL,FR.wav'), speakers=['FL', 'FR'])
    feedback.crop_heads()
    feedback.crop_tails()

    # Open feedforward measurement
    # Only FL-left and FR-right are needed here
    feedforward = HRIR(estimator)
    feedforward.open_recording(os.path.join(DIR_PATH, 'headphones.wav'), speakers=['FL', 'FR'])
    ffl = feedforward.irs['FL']['left'].frequency_response()
    ff_gain = ffl.center([100, 10000])
    zero = FrequencyResponse(name='zero', frequency=ffl.frequency, raw=np.zeros(ffl.frequency.shape))
    ffl.compensate(zero)
    ffl.smoothen_heavy_light()
    ffr = feedforward.irs['FR']['right'].frequency_response()
    ffr.raw += ff_gain
    ffr.compensate(zero)
    ffr.smoothen_heavy_light()
    feedforward_errors = {'left': ffl, 'right': ffr}

    # Open HRIR measurement
    hrir = HRIR(estimator)
    hrir.open_recording(os.path.join(DIR_PATH, 'FL,FR.wav'), speakers=['FL', 'FR'])
    hrir.crop_heads()
    hrir.crop_tails()
    fllfr = hrir.irs['FL']['left'].frequency_response()
    gain = fllfr.center([100, 10000])

    # Feedback vs HRIR
    fig, ax = plt.subplots(3, 2)
    fig.set_size_inches(18, 12)
    fig.suptitle('Feedback Compensation')
    i = 0
    feedback_errors = {'left': None, 'right': None}
    for speaker, pair in feedback.irs.items():
        j = 0
        for side, ir in pair.items():
            # HRIR is the target
            target = hrir.irs[speaker][side].frequency_response()
            target.raw += gain
            target.smoothen_fractional_octave(window_size=1/3, treble_window_size=1/3)

            # Frequency response of the headphone feedback measurement
            fr = ir.frequency_response()
            fr.raw += gain
            fr.error = fr.raw - target.raw
            fr.smoothen_heavy_light()
            # Add to this side average
            if feedback_errors[side] is None:
                feedback_errors[side] = fr.error_smoothed
            else:
                feedback_errors[side] += fr.error_smoothed

            # Plot
            ir.plot_fr(fr=fr, fig=fig, ax=ax[i, j])
            ax[i, j].set_title(f'{speaker}-{side}')
            ax[i, j].set_ylim([np.min(fr.error_smoothed), np.max(fr.error_smoothed)])

            j += 1
        i += 1

    for i, side in enumerate(['left', 'right']):
        feedback_errors[side] = FrequencyResponse(
            name=side,
            frequency=fllfr.frequency.copy(),
            error=feedback_errors[side] / 2
        )
        feedback_errors[side].plot_graph(fig=fig, ax=ax[2, i], show=False)

    sync_axes([ax[i, j] for i in range(ax.shape[0]) for j in range(ax.shape[1])])
    save_fig_as_png(os.path.join(DIR_PATH, 'feedback.png'), fig)

    # Feedforward
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(18, 9)
    fig.suptitle('Feedforward Compensation')
    ffl.plot_graph(fig=fig, ax=ax[0], show=False)
    ffr.plot_graph(fig=fig, ax=ax[1], show=False)
    save_fig_as_png(os.path.join(DIR_PATH, 'feedforward.png'), fig)

    # Feedback compensation vs Feedforward compensation
    feedback_errors['left'].raw = feedback_errors['left'].error
    fbg = feedback_errors['left'].center([200, 2000])
    feedback_errors['left'].error = feedback_errors['left'].raw
    feedback_errors['left'].raw = []
    feedback_errors['right'].error += fbg

    feedforward_errors['left'].raw = feedforward_errors['left'].error_smoothed
    ffg = feedforward_errors['left'].center([200, 2000])
    feedforward_errors['left'].error_smoothed = feedforward_errors['left'].raw
    feedforward_errors['left'].raw = []
    feedforward_errors['right'].error_smoothed += ffg

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(18, 9)
    fig.suptitle('Feedback vs Feedforward')
    for i, side in enumerate(['left', 'right']):
        config_fr_axis(ax[i])
        ax[i].plot(feedback_errors[side].frequency, feedback_errors[side].error)
        ax[i].plot(feedforward_errors[side].frequency, feedforward_errors[side].error_smoothed)
        difference = feedback_errors[side].error - feedforward_errors[side].error_smoothed
        ax[i].plot(feedback_errors[side].frequency, difference, color=COLORS['red'])
        ax[i].set_title(side)
        ax[i].legend(['Feedback', 'Feedforward', 'Difference'])
    save_fig_as_png(os.path.join(DIR_PATH, 'comparison.png'), fig)
    plt.show()


if __name__ == '__main__':
    main()
