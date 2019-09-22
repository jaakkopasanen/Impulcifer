# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from impulse_response import ImpulseResponse
from utils import sync_axes
from constants import SPEAKER_NAMES


def main(dir_path=None,
         test_signal=None,
         speakers=None,
         compensate_headphones=False,
         plot=False):
    """"""
    if dir_path is None or not os.path.isdir(dir_path):
        raise NotADirectoryError(f'Given dir path "{dir_path}"" is not a directory.')

    # Paths
    dir_path = os.path.abspath(dir_path)
    if not test_signal and os.path.isfile(os.path.join(dir_path, 'test.wav')):
        test_signal = os.path.join(dir_path, 'test.wav')
    recording = os.path.join(dir_path, 'recording.wav')
    headphones = os.path.join(dir_path, 'headphones.wav')

    # Read files
    estimator = ImpulseResponseEstimator.from_wav(test_signal)
    hrir = HRIR(estimator)

    if speakers is None:
        # Speakers not given, use multiple recording files with speaker names in the file names
        # Files must be of pattern FL,FR,BR.wav
        speaker_pattern = f'({"|".join(SPEAKER_NAMES + ["X"])})'
        pattern = r'^{speaker_pattern}+(,{speaker_pattern})*\.wav$'.format(speaker_pattern=speaker_pattern)
        for file_name in [f for f in os.listdir(dir_path) if re.match(pattern, f)]:
            # Read the speaker names from the file name into a list
            speakers = file_name.replace('.wav', '').split(',')
            # Form absolute path
            file_name = os.path.join(dir_path, file_name)
            # Open the file and add tracks to HRIR
            hrir.open_recording(file_name, speakers=speakers)
    else:
        # Speakers give, use recording.wav
        hrir.open_recording(recording, speakers=speakers)

    # Write multi-channel WAV file with sine sweeps for debugging
    hrir.write_wav(os.path.join(dir_path, 'responses.wav'))

    if plot:
        # Plot
        os.makedirs(os.path.join(dir_path, 'plots'))
        hrir.plot(dir_path=os.path.join(dir_path, 'plots'))

    # Crop noise and harmonics from the beginning
    hrir.crop_heads()

    # Crop noise from the tail
    hrir.crop_tails()

    if os.path.isfile(headphones):
        # Read WAV file
        hp_irs = HRIR(estimator)
        hp_irs.open_recording(headphones, speakers=['FL', 'FR'])
        hp_irs = [hp_irs.irs['FL']['left'], hp_irs.irs['FR']['right']]

        eq_irs = []
        biases = []
        frs = []
        for i, ir in enumerate(hp_irs):
            # Calculate magnitude response
            f, m = ir.magnitude_response()
            # Create frequency response
            name = 'Left' if i == 0 else 'Right'
            fr = FrequencyResponse(name=name, frequency=f[1:], raw=m[1:])
            fr.interpolate()

            # Take copy of the frequency response and
            fr_eq = fr.copy()
            # Centering removes bias and we want the bias to stay in the original
            fr_eq.center()
            # Aiming for flat response at the ear canal
            fr_eq.compensate(
                FrequencyResponse(name='zero', frequency=fr.frequency, raw=np.zeros(len(fr.frequency))),
                min_mean_error=False
            )
            # Smoothen
            fr_eq.smoothen_heavy_light()
            # Equalize to flat
            fr_eq.equalize(max_gain=15, treble_f_lower=20000, treble_f_upper=23000, treble_gain_k=1)

            # Copy equalization curve
            fr.equalization = fr_eq.equalization.copy()
            # Calculate equalized raw data
            fr.equalized_raw = fr.raw + fr.equalization
            frs.append(fr)
            # Create miniumum phase FIR filter
            eq_ir = fr.minimum_phase_impulse_response(fs=estimator.fs, f_res=10)
            eq_irs.append(ImpulseResponse(eq_ir, estimator.fs))
            # Calculate bias
            avg = np.mean(fr.equalized_raw[np.logical_and(fr.frequency >= 100, fr.frequency <= 10000)])
            biases.append(avg)

        # Balance equalization filters for left and right ear
        # Both ears need to have same level
        # Levels might be different due to measurement devices or microphone placement
        if biases[0] > biases[1]:
            # Left headphone measurement is louder, bring it down to level of right headphone
            eq_irs[0].data *= 10**((biases[1]-biases[0]) / 20)
            frs[0].equalization += biases[1] - biases[0]
            frs[0].equalized_raw += biases[1] - biases[0]
        else:
            # Right headphone measurement is louder, bring it down to level of left headphone
            eq_irs[1].data *= 10**((biases[0] - biases[1]) / 20)
            frs[1].equalization += biases[0] - biases[1]
            frs[1].equalized_raw += biases[0] - biases[1]

        if plot:
            # Headphone plots
            plots = {'left': {'fr': None, 'ir': None}, 'right': {'fr': None, 'ir': None}}
            for ir, fr, side in zip(eq_irs, frs, ['left', 'right']):
                fig, ax = plt.subplots(1, 2)
                fig.set_size_inches(15, 7)
                fr.plot_graph(fig=fig, ax=ax[0], show=False)
                ir.plot_ir(fig=fig, ax=ax[1], end=2e-3)
                plt.suptitle(f'Headphones {side}')
                plots[side]['fig'] = fig
                plots[side]['fr'] = ax[0]
                plots[side]['ir'] = ax[1]

            # Sync axes
            sync_axes([plots['left']['fr'], plots['right']['fr']])
            sync_axes([plots['left']['ir'], plots['right']['ir']])

            # Save headphone plots
            for side in ['left', 'right']:
                fig = plots[side]['fig']
                fig.savefig(os.path.join(dir_path, 'plots', f'Headphones {side}.png'))
                plt.close(fig)

        # Equalize HRIR with headphone compensation FIR filters
        for speaker, pair in hrir.irs.items():
            for side, ir in pair.items():
                if side == 'left':
                    hrir.irs[speaker][side].data = convolve(ir.data, eq_irs[0].data, mode='full')
                else:
                    hrir.irs[speaker][side].data = convolve(ir.data, eq_irs[1].data, mode='full')

    # Normalize gain
    hrir.normalize(target_db=0)

    # Write multi-channel WAV file with standard track order
    hrir.write_wav(os.path.join(dir_path, 'hrir.wav'))

    # Write multi-channel WAV file with HeSuVi track order
    hrir.write_wav(
        os.path.join(dir_path, 'hesuvi.wav'),
        track_order=['FL-left', 'FL-right', 'SL-left', 'SL-right', 'BL-left', 'BL-right', 'FC-left', 'FR-right',
                     'FR-left', 'SR-right', 'SR-left', 'BR-right', 'BR-left', 'FC-right']
    )


def create_cli():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dir_path', type=str, help='Path to directory for recordings and outputs.')
    arg_parser.add_argument('--test_signal', type=str, help='File path to sine sweep test signal.')
    arg_parser.add_argument('--speakers', type=str, default=argparse.SUPPRESS,
                            help='Order of speakers in the recording as a comma separated list of speaker channel '
                                 'names. Supported names are "FL" (front left), "FR" (front right), '
                                 '"FC" (front center), "BL" (back left), "BR" (back right), '
                                 '"SL" (side left), "SR" (side right)". For example: "FL,FR".')
    arg_parser.add_argument('--plot', action='store_true', help='Plot graphs for debugging.')
    args = vars(arg_parser.parse_args())
    if 'speakers' in args and args['speakers'] is not None:
        args['speakers'] = args['speakers'].upper().split(',')
    return args


if __name__ == '__main__':
    main(**create_cli())
