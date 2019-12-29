# -*- coding: utf-8 -*-

import os
import re
import argparse
from tabulate import tabulate
from datetime import datetime
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from utils import sync_axes, read_wav, save_fig_as_png
from constants import SPEAKER_NAMES, SPEAKER_LIST_PATTERN, IR_ROOM_SPL


def main(dir_path=None,
         test_signal=None,
         room_target=None,
         room_mic_calibration=None,
         fs=None,
         plot=False,
         channel_balance=None,
         do_room_correction=True,
         do_headphone_compensation=True,
         do_equalization=True):
    """"""
    if dir_path is None or not os.path.isdir(dir_path):
        raise NotADirectoryError(f'Given dir path "{dir_path}"" is not a directory.')

    # Dir path as absolute
    dir_path = os.path.abspath(dir_path)

    # Impulse response estimator
    if not test_signal:
        # Test signal not explicitly given, try Pickle first then WAV
        if os.path.isfile(os.path.join(dir_path, 'test.pkl')):
            test_signal = os.path.join(dir_path, 'test.pkl')
        elif os.path.isfile(os.path.join(dir_path, 'test.wav')):
            test_signal = os.path.join(dir_path, 'test.wav')
    if re.match(r'^.+\.wav$', test_signal, flags=re.IGNORECASE):
        # Test signal is WAV file
        estimator = ImpulseResponseEstimator.from_wav(test_signal)
    elif re.match(r'^.+\.pkl$', test_signal, flags=re.IGNORECASE):
        # Test signal is Pickle file
        estimator = ImpulseResponseEstimator.from_pickle(test_signal)
    else:
        raise TypeError(f'Unknown file extension for test signal "{test_signal}"')

    # Headphone compensation frequency responses
    hp_left = None
    hp_right = None
    if do_headphone_compensation:
        hp_left, hp_right = headphone_compensation(estimator, dir_path)

    # Room correction frequency responses
    room_frs = None
    if do_room_correction:
        _, room_frs = correct_room(
            estimator, dir_path,
            target=room_target,
            mic_calibration=room_mic_calibration,
            plot=plot
        )

    # Equalization
    eq_left = None
    eq_right = None
    if do_equalization:
        eq_left, eq_right = read_eq(estimator, dir_path)

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

    # Write multi-channel WAV file with sine sweeps for debugging
    hrir.write_wav(os.path.join(dir_path, 'responses.wav'))

    # Equalize all
    if do_headphone_compensation or do_room_correction or do_equalization:
        for speaker, pair in hrir.irs.items():
            for side, ir in pair.items():
                fr = FrequencyResponse(name='eq')
                fr.raw = np.zeros(fr.frequency.shape)
                fr.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)
                fr.error = np.zeros(fr.frequency.shape)

                if room_frs is not None and speaker in room_frs and side in room_frs[speaker]:
                    # Room correction
                    fr.error += room_frs[speaker][side].error

                if side == 'left':
                    hp_eq = hp_left
                    eq = eq_left
                else:
                    hp_eq = hp_right
                    eq = eq_right

                if hp_eq is not None:
                    # Headphone compensation
                    fr.error += hp_eq.error
                if eq is not None and type(eq) == FrequencyResponse:
                    # Equalization
                    fr.error += eq.error

                # Smoothen and equalize
                fr.smoothen_heavy_light()
                fr.equalize(max_gain=30, treble_f_lower=10000, treble_f_upper=estimator.fs / 2)

                # Create FIR filter and equalize
                fir = fr.minimum_phase_impulse_response(fs=estimator.fs, normalize=False)
                ir.equalize(fir)

                if fr is not None and type(fr) == np.ndarray:
                    # Equalization filter as FIR filter in WAV file
                    ir.equalize(fr)

    # Correct channel balance
    if channel_balance is not None:
        hrir.correct_channel_balance(channel_balance)

    # Normalize gain
    hrir.normalize(target_db=0)

    if plot:
        # Convolve test signal, re-plot waveform and spectrogram
        for speaker, pair in hrir.irs.items():
            for side, ir in pair.items():
                ir.recording = signal.convolve(estimator.test_signal, ir.data, mode='full')
        # Plot post processing
        hrir.plot(os.path.join(dir_path, 'plots', 'post'))

    # Plot results, always
    hrir.plot_result(os.path.join(dir_path, 'plots'))

    # Re-sample
    if fs is not None and fs != hrir.fs:
        hrir.resample(fs)

    # Write multi-channel WAV file with standard track order
    hrir.write_wav(os.path.join(dir_path, 'hrir.wav'))

    # Write multi-channel WAV file with HeSuVi track order
    hrir.write_wav(
        os.path.join(dir_path, 'hesuvi.wav'),
        track_order=['FL-left', 'FL-right', 'SL-left', 'SL-right', 'BL-left', 'BL-right', 'FC-left', 'FR-right',
                     'FR-left', 'SR-right', 'SR-left', 'BR-right', 'BR-left', 'FC-right']
    )


def read_eq(estimator, dir_path):
    """Reads equalization FIR filter or CSV settings

    Args:
        estimator: ImpulseResponseEstimator
        dir_path: Path to directory

    Returns:
        - Left side FIR as Numpy array or FrequencyResponse or None
        - Right side FIR as Numpy array or FrequencyResponse or None
    """
    # FIR filter as WAV file
    eq_path = os.path.join(dir_path, 'eq.wav')
    if os.path.isfile(eq_path):
        eq_fs, firs = read_wav(eq_path, expand=True)
        if eq_fs != estimator.fs:
            raise ValueError('Equalization FIR filter sampling rate must match HRIR sampling rate.')
        return firs[0], firs[1]

    # Equalization frequency responses as CSV files

    # Default for both sides
    eq_path = os.path.join(dir_path, 'eq.csv')
    eq_fr = None
    if os.path.isfile(eq_path):
        eq_fr = FrequencyResponse.read_from_csv(eq_path)

    # Left
    left_path = os.path.join(dir_path, 'eq-left.csv')
    left_fr = None
    if os.path.isfile(left_path):
        left_fr = FrequencyResponse.read_from_csv(left_path)
    elif eq_fr is not None:
        left_fr = eq_fr
    if left_fr is not None:
        left_fr.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)

    # Right
    right_path = os.path.join(dir_path, 'eq-right.csv')
    right_fr = None
    if os.path.isfile(right_path):
        right_fr = FrequencyResponse.read_from_csv(right_path)
    elif eq_fr is not None:
        right_fr = eq_fr
    if right_fr is not None:
        right_fr.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)

    return left_fr, right_fr


def correct_room(estimator, dir_path, target=None, mic_calibration=None, plot=False):
    """Corrects room acoustics

    Args:
        estimator: ImpulseResponseEstimator
        dir_path: Path to directory
        target: Path to room target response CSV file
        mic_calibration: Path to room measurement microphone calibration file. AutoEQ CSV files and MiniDSP'
                              txt files are  supported.
        plot: Plot graphs?

    Returns:
        - Room Impulse Responses as HRIR or None
        - Equalization frequency responses as dict of dicts (similar to HRIR) or None
    """
    # Read room measurement files
    rir = HRIR(estimator)
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
        return None, None

    # Room target
    if target is None:
        target = os.path.join(dir_path, 'room-target.csv')
    if os.path.isfile(target):
        # File exists, create frequency response
        target = FrequencyResponse.read_from_csv(target)
        target.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)
        target.center()
    else:
        # No room target specified, use flat
        target = FrequencyResponse(name='room-target')
        target.raw = np.zeros(target.frequency.shape)
        target.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)

    # Room measurement microphone calibration
    if mic_calibration is None:
        # Room mic calibration file path not given, try csv first then txt
        mic_calibration = os.path.join(dir_path, 'room-mic-calibration.csv')
        if not os.path.isfile(mic_calibration):
            mic_calibration = os.path.join(dir_path, 'room-mic-calibration.txt')
    elif not os.path.isfile(mic_calibration):
        # Room mic calibration file path given, but the file doesn't exist
        raise FileNotFoundError(f'Room mic calibration file doesn\'t exist at "{mic_calibration}"')
    if os.path.isfile(mic_calibration):
        # File found, create frequency response
        mic_calibration = FrequencyResponse.read_from_csv(mic_calibration)
        mic_calibration.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)
        mic_calibration.center()
    else:
        # File not found, skip calibration
        mic_calibration = None

    # Crop heads and tails from room impulse responses
    for speaker, pair in rir.irs.items():
        for side, ir in pair.items():
            ir.crop_head()
    rir.crop_tails()
    rir.write_wav(os.path.join(dir_path, 'room-responses.wav'))

    figs = None
    if plot:
        # Plot all but frequency response
        plot_dir = os.path.join(dir_path, 'plots', 'room')
        os.makedirs(plot_dir, exist_ok=True)
        figs = rir.plot(plot_fr=False, close_plots=False)

    # Create equalization frequency responses
    reference_gain = None
    fr_axes = []
    frs = dict()
    for speaker, pair in rir.irs.items():
        frs[speaker] = dict()
        for side, ir in pair.items():
            # Create frequency response
            fr = ir.frequency_response()
            fr.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)

            if mic_calibration is not None:
                # Calibrate frequency response
                fr.raw -= mic_calibration.raw

            # Process
            fr.compensate(target, min_mean_error=True)

            # Sync gains
            if reference_gain is None:
                reference_gain = fr.center([100, 10000])  # Shifted (up) by this many dB
            else:
                fr.raw += reference_gain

            # Adjust target level with the (negative) gain caused by speaker-ear distance in reverberant room
            target_adjusted = target.copy()
            target_adjusted.raw += IR_ROOM_SPL[speaker][side]
            # Compensate with the adjusted room target
            fr.compensate(target_adjusted, min_mean_error=False)

            # Add frequency response
            frs[speaker][side] = fr

            if plot:
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                fr = fr.copy()
                fr.smoothen_heavy_light()
                _, fr_ax = ir.plot_fr(
                    fr=fr,
                    fig=figs[speaker][side],
                    ax=figs[speaker][side].get_axes()[4],
                    plot_raw=False,
                    plot_error=False,
                    plot_file_path=file_path,
                    fix_ylim=True
                )
                fr_axes.append(fr_ax)

    if plot:
        # Sync FR plot axes
        sync_axes(fr_axes)
        # Save figures
        for speaker, pair in figs.items():
            for side, fig in pair.items():
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                os.makedirs(os.path.split(file_path)[0], exist_ok=True)
                save_fig_as_png(file_path, fig)
                plt.close(fig)

    return rir, frs


def headphone_compensation(estimator, dir_path):
    """Equalizes HRIR tracks with headphone compensation measurement.

    Args:
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to output directory

    Returns:
        None
    """
    # Read WAV file
    hp_irs = HRIR(estimator)
    hp_irs.open_recording(os.path.join(dir_path, 'headphones.wav'), speakers=['FL', 'FR'])
    hp_irs.write_wav(os.path.join(dir_path, 'headphone-responses.wav'))

    # Frequency responses
    left = hp_irs.irs['FL']['left'].frequency_response()
    right = hp_irs.irs['FR']['right'].frequency_response()

    # Center by left channel
    gain = left.center([100, 10000])
    right.raw += gain

    # Compensate
    zero = FrequencyResponse(name='zero', frequency=left.frequency, raw=np.zeros(len(left.frequency)))
    left.compensate(zero, min_mean_error=False)
    right.compensate(zero, min_mean_error=False)

    # Headphone plots
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    fig.set_size_inches(22, 10)
    fig.suptitle('Headphones')

    # Left
    axl = fig.add_subplot(gs[0, 0])
    left.plot_graph(fig=fig, ax=axl, show=False)
    axl.set_title('Left')
    # Right
    axr = fig.add_subplot(gs[1, 0])
    right.plot_graph(fig=fig, ax=axr, show=False)
    axr.set_title('Right')
    # Sync axes
    sync_axes([axl, axr])

    # Combined
    _left = left.copy()
    _right = right.copy()
    _left.center([100, 10000])
    _right.center([100, 10000])
    ax = fig.add_subplot(gs[:, 1:])
    ax.plot(_left.frequency, _left.raw, linewidth=1, color='#1f77b4')
    ax.plot(_right.frequency, _right.raw, linewidth=1, color='#d62728')
    ax.plot(_left.frequency, _left.raw - _right.raw, linewidth=1, color='#680fb9')
    ax.set_title('Comparison')
    ax.legend(['Left raw', 'Right raw', 'Difference'], fontsize=8)
    ax.set_xlabel('Frequency (Hz)')
    ax.semilogx()
    ax.set_xlim([20, 20000])
    ax.set_ylabel('Amplitude (dBr)')
    ax.grid(True, which='major')
    ax.grid(True, which='minor')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # Save headphone plots
    file_path = os.path.join(dir_path, 'plots', 'headphones.png')
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    save_fig_as_png(file_path, fig)
    plt.close(fig)

    return left, right


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
            reverb_time = ir.reverberation_time(target_level=-60)
            table.append([
                speaker,
                side,
                f'{ir.pnr():.1f} dB',
                f'{_itd:.1f} us',
                f'{ir.active_duration() * 1000:.1f} ms',
                f'{reverb_time * 1000:.1f} ms' if reverb_time is not None else '-'
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
    arg_parser.add_argument('--dir_path', type=str, required=True, help='Path to directory for recordings and outputs.')
    arg_parser.add_argument('--test_signal', type=str, default=argparse.SUPPRESS,
                            help='Path to sine sweep test signal or pickled impulse response estimator.')
    arg_parser.add_argument('--room_target', type=str, default=argparse.SUPPRESS,
                            help='Path to room target response AutoEQ style CSV file.')
    arg_parser.add_argument('--room_mic_calibration', type=str, default=argparse.SUPPRESS,
                            help='Path to room measurement microphone calibration file.')
    arg_parser.add_argument('--no_room_correction', action='store_false', dest='do_room_correction',
                            help='Skip room correction.')
    arg_parser.add_argument('--no_headphone_compensation', action='store_false', dest='do_headphone_compensation',
                            help='Skip headphone compensation.')
    arg_parser.add_argument('--no_equalization', action='store_false', dest='do_equalization',
                            help='Skip equalization.')
    arg_parser.add_argument('--fs', type=int, default=argparse.SUPPRESS, help='Output sampling rate in Hertz.')
    arg_parser.add_argument('--plot', action='store_true', help='Plot graphs for debugging.')
    arg_parser.add_argument('--channel_balance', type=str, default=argparse.SUPPRESS,
                            help='Channel balance correction by equalizing left and right ear results to the same '
                                 'level or frequency response. "trend" equalizes right side by the difference trend '
                                 'of right and left side. "left" equalizes right side to left side fr, "right" '
                                 'equalizes left side to right side fr, "avg" equalizes both to the average fr, "min" '
                                 'equalizes both to the minimum of left and right side frs. Number values will boost '
                                 'or attenuate right side relative to left side by the number of dBs. "mids" is the '
                                 'same as the numerical values but guesses the value automatically from mid frequency '
                                 'levels.')
    args = vars(arg_parser.parse_args())
    return args


if __name__ == '__main__':
    main(**create_cli())
