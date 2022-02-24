# -*- coding: utf-8 -*-

import os
import sys
from glob import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from autoeq.frequency_response import FrequencyResponse
DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.insert(1, os.path.dirname(os.path.dirname(DIR_PATH)))  # changed to also work in conda Python environments.
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from utils import optimize_png_size



def main(test_signal):
    estimator = ImpulseResponseEstimator.from_pickle(test_signal)

    # Room mic calibration
    room_mic_calibration = os.path.join(DIR_PATH, 'room-mic-calibration.csv')
    if not os.path.isfile(room_mic_calibration):
        room_mic_calibration = os.path.join(DIR_PATH, 'room-mic-calibration.txt')
    if os.path.isfile(room_mic_calibration):
        # File found, create frequency response
        room_mic_calibration = FrequencyResponse.read_from_csv(room_mic_calibration)
        room_mic_calibration.interpolate(f_step=1.01, f_min=10, f_max=20e3)
        room_mic_calibration.center()
    else:
        room_mic_calibration = None

    # Room measurement mic
    rooms = []
    for file_path in glob(os.path.join(DIR_PATH, 'room*.wav')):
        room = HRIR(estimator)
        room.open_recording(file_path, speakers=['FL'], side='left')
        fr = room.irs['FL']['left'].frequency_response()
        fr.interpolate(f_step=1.01, f_min=10, f_max=20e3)
        rooms.append(fr)
        if room_mic_calibration is not None:
            # Adjust by calibration data
            rooms[-1].raw -= room_mic_calibration.raw

    # Binaural mics
    lefts = []
    rights = []
    for file_path in glob(os.path.join(DIR_PATH, 'binaural*.wav')):
        binaural = HRIR(estimator)
        binaural.open_recording(file_path, speakers=['FL'])
        lefts.append(binaural.irs['FL']['left'].frequency_response())
        rights.append(binaural.irs['FL']['right'].frequency_response())

    # Setup plot
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 9)
    ax.set_title('Microphone calibration')
    ax.set_xlabel('Frequency (Hz)')
    ax.semilogx()
    ax.set_xlim([20, 20e3])
    ax.set_ylabel('Amplitude (dB)')
    ax.grid(True, which='major')
    ax.grid(True, which='minor')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # Room measurement mic
    room = FrequencyResponse(
        name='Room measurement mic',
        frequency=rooms[0].frequency,
        raw=np.mean(np.vstack([x.raw for x in rooms]), axis=0)
    )
    room.interpolate(f_step=1.01, f_min=10, f_max=20e3)
    room.smoothen_fractional_octave(window_size=1/6, treble_window_size=1/6)
    room.raw = room.smoothed.copy()
    room.smoothed = []
    room.center([60, 10000])
    ax.plot(room.frequency, room.raw, color='#680fb9', linewidth=0.5)

    # Left binaural mic
    left = FrequencyResponse(
        name='Left binaural mic',
        frequency=lefts[0].frequency,
        raw=np.mean(np.vstack([x.raw for x in lefts]), axis=0)
    )
    left.interpolate(f_step=1.01, f_min=10, f_max=20e3)
    left.smoothen_fractional_octave(window_size=1/6, treble_window_size=1/6)
    left.raw = left.smoothed.copy()
    left.smoothed = []
    gain = left.center([60, 10000])
    ax.plot(left.frequency, left.raw, color='#7db4db', linewidth=0.5)
    ax.plot(left.frequency, left.raw - room.raw, color='#1f77b4')
    left.write_to_csv(os.path.join(DIR_PATH, 'left-mic-calibration.csv'))

    # Right binaural mic
    right = FrequencyResponse(
        name='Right binaural mic',
        frequency=rights[0].frequency,
        raw=np.mean(np.vstack([x.raw for x in rights]), axis=0)
    )
    right.interpolate(f_step=1.01, f_min=10, f_max=20e3)
    right.smoothen_fractional_octave(window_size=1/6, treble_window_size=1/6)
    right.raw = right.smoothed.copy()
    right.smoothed = []
    right.raw += gain
    ax.plot(right.frequency, right.raw, color='#dd8081', linewidth=0.5)
    ax.plot(right.frequency, right.raw - room.raw, color='#d62728')
    right.write_to_csv(os.path.join(DIR_PATH, 'right-mic-calibration.csv'))

    ax.legend(['Room', 'Left', 'Left calibration', 'Right', 'Right calibration'])

    # Save figure
    file_path = os.path.join(DIR_PATH, 'Results.png')
    fig.savefig(file_path, bbox_inches='tight')
    optimize_png_size(file_path)

    plt.show()


def create_cli():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--test_signal', type=str, required=True,
                            help='Path to sine sweep test signal or pickled impulse response estimator.')
    cli_args = arg_parser.parse_args()
    return vars(cli_args)


if __name__ == '__main__':
    main(**create_cli())
