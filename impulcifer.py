# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tabulate import tabulate
from datetime import datetime
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from impulse_response import ImpulseResponse
from utils import sync_axes, read_wav
from constants import SPEAKER_NAMES


def main(dir_path=None,
         test_signal=None,
         fs=None,
         plot=False):
    """"""
    if dir_path is None or not os.path.isdir(dir_path):
        raise NotADirectoryError(f'Given dir path "{dir_path}"" is not a directory.')

    # Paths
    dir_path = os.path.abspath(dir_path)
    if not test_signal:
        if os.path.isfile(os.path.join(dir_path, 'test.pkl')):
            test_signal = os.path.join(dir_path, 'test.pkl')
        elif os.path.isfile(os.path.join(dir_path, 'test.wav')):
            test_signal = os.path.join(dir_path, 'test.wav')
    headphones = os.path.join(dir_path, 'headphones.wav')
    eq = os.path.join(dir_path, 'eq.wav')

    # Read files
    if re.match('^.+\.wav$', test_signal, flags=re.IGNORECASE):
        estimator = ImpulseResponseEstimator.from_wav(test_signal)
    elif re.match('^.+\.pkl$', test_signal, flags=re.IGNORECASE):
        estimator = ImpulseResponseEstimator.from_pickle(test_signal)
    else:
        raise TypeError(f'Unknown file extension for test signal "{test_signal}"')
    hrir = HRIR(estimator)

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

    # Write multi-channel WAV file with sine sweeps for debugging
    hrir.write_wav(os.path.join(dir_path, 'responses.wav'))

    # Write info and stats in readme
    write_readme(os.path.join(dir_path, 'README.md'), hrir, fs)

    if plot:
        # Plot
        os.makedirs(os.path.join(dir_path, 'plots'), exist_ok=True)
        hrir.plot(dir_path=os.path.join(dir_path, 'plots'))

    # Crop noise and harmonics from the beginning
    hrir.crop_heads()

    # Crop noise from the tail
    hrir.crop_tails()

    # Compensate headphones
    if os.path.isfile(headphones):
        fig_path = os.path.join(dir_path, 'plots', f'Headphones.png') if plot else None
        compensate_headphones(headphones, hrir, fig_path=fig_path)

    # Apply given equalization filter
    if os.path.isfile(eq):
        fs, firs = read_wav(eq)
        hrir.equalize(firs)

    # Resample
    if fs is not None and fs != hrir.fs:
        hrir.resample(fs)

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


def compensate_headphones(recording, hrir, fig_path=None):
    """Equalizes HRIR tracks with headphone compensation measurement.

    Args:
        recording: File path to sine sweep recording made with headphones
        hrir: HRIR instance for the speaker measurements
        fig_path: File path for saving graphs

    Returns:
        None
    """
    # Read WAV file
    hp_irs = HRIR(hrir.estimator)
    hp_irs.open_recording(recording, speakers=['FL', 'FR'])
    hp_irs = [hp_irs.irs['FL']['left'], hp_irs.irs['FR']['right']]

    firs = []
    biases = []
    frs = []
    for i, ir in enumerate(hp_irs):
        # Calculate magnitude response
        f, m = ir.magnitude_response()
        # Create frequency response
        name = 'Left' if i == 0 else 'Right'
        n = ir.fs / 2 / 4  # 4 Hz resolution
        step = int(len(f) / n)
        fr = FrequencyResponse(name=name, frequency=f[1::step], raw=m[1::step])
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
        # Create minimum phase FIR filter
        eq_ir = fr.minimum_phase_impulse_response(fs=hrir.fs, f_res=10)
        firs.append(ImpulseResponse(eq_ir, hrir.fs))
        # Calculate bias
        avg = np.mean(fr.equalized_raw[np.logical_and(fr.frequency >= 100, fr.frequency <= 10000)])
        biases.append(avg)

    # Balance equalization filters for left and right ear
    # Both ears need to have same level
    # Levels might be different due to measurement devices or microphone placement
    if biases[0] > biases[1]:
        # Left headphone measurement is louder, bring it down to level of right headphone
        firs[0].data *= 10 ** ((biases[1] - biases[0]) / 20)
        frs[0].equalization += biases[1] - biases[0]
        frs[0].equalized_raw += biases[1] - biases[0]
    else:
        # Right headphone measurement is louder, bring it down to level of left headphone
        firs[1].data *= 10 ** ((biases[0] - biases[1]) / 20)
        frs[1].equalization += biases[0] - biases[1]
        frs[1].equalized_raw += biases[0] - biases[1]

    if fig_path is not None:
        # Headphone plots
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(15, 7)
        fig.suptitle('Headphones')
        for i, side in enumerate(['Left', 'Right']):
            frs[i].plot_graph(fig=fig, ax=ax[i], show=False)
            ax[i].set_title(side)
        # Sync axes
        sync_axes([ax[0], ax[1]])
        # Save headphone plots
        fig.savefig(fig_path)
        plt.close(fig)

    # Equalize HRIR with headphone compensation FIR filters
    hrir.equalize(firs)


def write_readme(file_path, hrir, fs):
    """Writes info and stats to readme file.

    Args:
        file_path: Path to readme file
        hrir: HRIR instance
        fs: Output sampling rate

    Returns:
        Readme string
    """
    if fs is None:
        fs = hrir.fs

    table = []
    speaker_names = sorted(hrir.irs.keys(), key=lambda x: SPEAKER_NAMES.index(x))
    for speaker in speaker_names:
        pair = hrir.irs[speaker]
        itd = np.abs(pair['right'].peak_index() - pair['left'].peak_index()) / hrir.fs * 1e6
        for side, ir in pair.items():
            # Zero for the first ear
            _itd = itd if side == 'left' and speaker[1] == 'R' or side == 'right' and speaker[1] == 'L' else 0.0
            table.append([
                speaker,
                side,
                f'{ir.pnr():.1f} dB',
                f'{_itd:.1f} us',
                f'{ir.active_duration() * 1000:.1f} ms',
                f'{ir.reverberation_time(target_level=-60) * 1000:.1f} ms'
            ])
    table_str = tabulate(
        table,
        headers=['Speaker', 'Side', 'PNR', 'ITD', 'Length', 'RT60'],
        tablefmt='github'
    )
    s = f'''# HRIR

    **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
    **Input sampling rate:** {hrir.fs} Hz  
    **Output sampling rate:** {fs} Hz  

    {table_str}
    '''
    s = re.sub('\n[ \t]+', '\n', s).strip()

    with open(file_path, 'w') as f:
        f.write(s)

    return s


def create_cli():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dir_path', type=str, help='Path to directory for recordings and outputs.')
    arg_parser.add_argument('--test_signal', type=str,
                            help='Path to sine sweep test signal or pickled impulse response estimator.')
    arg_parser.add_argument('--fs', type=int, help='Output sampling rate in Hertz.')
    arg_parser.add_argument('--plot', action='store_true', help='Plot graphs for debugging.')
    args = vars(arg_parser.parse_args())
    if 'speakers' in args and args['speakers'] is not None:
        args['speakers'] = args['speakers'].upper().split(',')
    return args


if __name__ == '__main__':
    main(**create_cli())
