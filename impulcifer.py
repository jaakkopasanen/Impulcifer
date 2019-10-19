# -*- coding: utf-8 -*-

import os
import re
import argparse
from tabulate import tabulate
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from impulse_response import ImpulseResponse
from utils import sync_axes, read_wav
from constants import SPEAKER_NAMES, SPEAKER_LIST_PATTERN, IR_ROOM_SPL


def main(dir_path=None,
         test_signal=None,
         room_target=None,
         room_mic_calibration=None,
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
    if re.match(r'^.+\.wav$', test_signal, flags=re.IGNORECASE):
        estimator = ImpulseResponseEstimator.from_wav(test_signal)
    elif re.match(r'^.+\.pkl$', test_signal, flags=re.IGNORECASE):
        estimator = ImpulseResponseEstimator.from_pickle(test_signal)
    else:
        raise TypeError(f'Unknown file extension for test signal "{test_signal}"')

    # HRIR measurements
    hrir = HRIR(estimator)
    pattern = r'^{pattern}\.wav$'.format(pattern=SPEAKER_LIST_PATTERN)  # FL,FR.wav
    for file_name in [f for f in os.listdir(dir_path) if re.match(pattern, f)]:
        # Read the speaker names from the file name into a list
        speakers = re.search(SPEAKER_LIST_PATTERN, file_name)[0].split(',')
        # Form absolute path
        file_path = os.path.join(dir_path, file_name)
        # Open the file and add tracks to HRIR
        hrir.open_recording(file_path, speakers=speakers)
    if len(hrir.irs) == 0:
        raise ValueError('No HRIR recordings found in the directory.')

    # Write multi-channel WAV file with sine sweeps for debugging
    hrir.write_wav(os.path.join(dir_path, 'responses.wav'))

    # Write info and stats in readme
    write_readme(os.path.join(dir_path, 'README.md'), hrir, fs)

    if plot:
        # Plot graphs pre processing
        os.makedirs(os.path.join(dir_path, 'plots', 'pre'), exist_ok=True)
        hrir.plot(dir_path=os.path.join(dir_path, 'plots', 'pre'))

    # Crop noise and harmonics from the beginning
    hrir.crop_heads()

    # Crop noise from the tail
    hrir.crop_tails()

    # Room correction
    rir = correct_room(
        hrir,
        dir_path=dir_path,
        room_target=room_target,
        room_mic_calibration=room_mic_calibration,
        plot=plot
    )
    # TODO: Add room stats to README

    # Compensate headphones
    if os.path.isfile(headphones):
        fig_path = os.path.join(dir_path, 'plots', f'Headphones.png') if plot else None
        compensate_headphones(headphones, hrir, fig_path=fig_path)

    # Apply given equalization filter
    if os.path.isfile(eq):
        eq_fs, firs = read_wav(eq)
        if eq_fs != hrir.fs:
            raise ValueError('Equalization FIR filter sampling rate must match HRIR sampling rate.')
        hrir.equalize(firs)

    # Re-sample
    if fs is not None and fs != hrir.fs:
        hrir.resample(fs)

    # Normalize gain
    hrir.normalize(target_db=0)

    if plot:
        # Plot post processing
        # TODO: Convolve test signal, re-plot waveform and spectrogram
        hrir.plot(os.path.join(dir_path, 'plots', 'post'))
        # Plot results
        hrir.plot_result(os.path.join(dir_path, 'plots'))

    # Write multi-channel WAV file with standard track order
    hrir.write_wav(os.path.join(dir_path, 'hrir.wav'))

    # Write multi-channel WAV file with HeSuVi track order
    hrir.write_wav(
        os.path.join(dir_path, 'hesuvi.wav'),
        track_order=['FL-left', 'FL-right', 'SL-left', 'SL-right', 'BL-left', 'BL-right', 'FC-left', 'FR-right',
                     'FR-left', 'SR-right', 'SR-left', 'BR-right', 'BR-left', 'FC-right']
    )


def correct_room(hrir, dir_path=None, room_target=None, room_mic_calibration=None, plot=False):
    """Corrects room acoustics

    Args:
        hrir: HRIR
        dir_path: Path to output directory
        room_target: Path to room target response CSV file
        room_mic_calibration: Path to room measurement microphone calibration file. AutoEQ CSV files and MiniDSP'
                              txt files are  supported.
        plot: Plot graphs?

    Returns:
        Room Impulse Responses as HRIR or None
    """
    # Read room measurement files
    rir = HRIR(hrir.estimator)
    # room-BL,SL.wav, room-left-FL,FR.wav, room-right-FC.wav, etc...
    pattern = r'^room-{pattern}(-(left|right))?\.wav$'.format(pattern=SPEAKER_LIST_PATTERN)
    for i, file_name in enumerate([f for f in os.listdir(dir_path) if re.match(pattern, f)]):
        # Read the speaker names from the file name into a list
        speakers = re.search(SPEAKER_LIST_PATTERN, file_name)
        if speakers is not None:
            speakers = speakers[0].split(',')
        # Read side if present
        side = re.search(r'(left|right)', file_name)[0]
        # Form absolute path
        file_path = os.path.join(dir_path, file_name)
        # Read file
        rir.open_recording(file_path, speakers, side=side)
    if not len(rir.irs):
        # No room recording files found
        return None

    # Room target
    if room_target is None:
        room_target = os.path.join(dir_path, 'room-target.csv')
        if os.path.isfile(room_target):
            # File exists, create frequency response
            room_target = FrequencyResponse.read_from_csv(room_target)
            room_target.interpolate()
            room_target.center()
        else:
            # No room target specified, use flat
            room_target = FrequencyResponse(name='room-target')
            room_target.raw = np.zeros(room_target.frequency.shape)

    # Room measurement microphone calibration
    if room_mic_calibration is None:
        # TODO: quess CSV format
        room_mic_calibration = os.path.join(dir_path, 'room-mic-calibration.csv')
        if os.path.isfile(room_mic_calibration):
            # File found, create frequency response
            room_mic_calibration = FrequencyResponse.read_from_csv(room_mic_calibration)
            room_mic_calibration.interpolate()
            room_mic_calibration.center()
        else:
            room_mic_calibration = None

    # Crop heads and tails from room impulse responses
    for speaker, pair in rir.irs.items():
        for side, ir in pair.items():
            ir.crop_head()
    rir.crop_tails()
    rir.write_wav(os.path.join(dir_path, 'room-responses.wav'))

    figs = None
    if plot:
        plot_dir = os.path.join(dir_path, 'plots', 'room')
        # Plot all but frequency response
        os.makedirs(plot_dir, exist_ok=True)
        figs = rir.plot(plot_fr=False, close_plots=False)

    # Create equalization filters and equalize
    ref_gain = None
    for speaker, pair in rir.irs.items():
        for side, ir in pair.items():
            fr = ir.frequency_response()
            if room_mic_calibration is not None:
                # Calibrate frequency response
                fr.raw -= room_mic_calibration.raw

            # Save original data for gain syncing
            original = fr.raw.copy()

            # Process
            fr.process(
                compensation=room_target,
                min_mean_error=True,
                equalize=True,
                max_gain=30,
                treble_f_lower=10000,
                treble_f_upper=20000
            )

            # Sync gains
            sl = np.logical_and(fr.frequency >= 100, fr.frequency <= 10000)
            gain = np.mean(original[sl] + fr.equalization[sl])
            if ref_gain is None:
                ref_gain = gain
            fir_gain = 10**((ref_gain - gain) / 20)

            # TODO: Some alien-tech mixed phase filter
            fir = fr.minimum_phase_impulse_response()
            fir *= fir_gain
            # Add SPL change from distance
            fir *= 10**(IR_ROOM_SPL[speaker][side] / 20)
            # Equalize
            hrir.irs[speaker][side].equalize(fir)

            if plot:
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                ir.plot_fr(
                    fr=fr,
                    fig=figs[speaker][side],
                    ax=figs[speaker][side].get_axes()[4],
                    plot_raw=True,
                    plot_error=False,
                    plot_file_path=file_path
                )

    if plot:
        # Sync FR plot axes
        axes = []
        for speaker, pair in figs.items():
            for side, fig in pair.items():
                axes.append(fig.get_axes()[5])
        sync_axes(axes)
        # Save figures
        for speaker, pair in figs.items():
            for side, fig in pair.items():
                fig.savefig(os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png'))
                plt.close(fig)

    return rir


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
    arg_parser.add_argument('--room_target', type=str,
                            help='Path to room target response AutoEQ style CSV file.')
    arg_parser.add_argument('--room_mic_calibration', type=str,
                            help='Path to room measurement microphone calibration file. AutoEQ CSV files and MiniDSP'
                                 'txt files are  supported.')
    arg_parser.add_argument('--fs', type=int, help='Output sampling rate in Hertz.')
    arg_parser.add_argument('--plot', action='store_true', help='Plot graphs for debugging.')
    args = vars(arg_parser.parse_args())
    if 'speakers' in args and args['speakers'] is not None:
        args['speakers'] = args['speakers'].upper().split(',')
    return args


if __name__ == '__main__':
    main(**create_cli())
