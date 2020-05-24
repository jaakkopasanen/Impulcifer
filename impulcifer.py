# -*- coding: utf-8 -*-

import os
import re
import argparse
from tabulate import tabulate
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from room_correction import room_correction
from utils import sync_axes, save_fig_as_png
from constants import SPEAKER_NAMES, SPEAKER_LIST_PATTERN, HESUVI_TRACK_ORDER


def main(dir_path=None,
         test_signal=None,
         room_target=None,
         room_mic_calibration=None,
         fs=None,
         plot=False,
         channel_balance=None,
         target_level=None,
         fr_combination_method='average',
         specific_limit=20000,
         generic_limit=1000,
         bass_boost_gain=0.0,
         bass_boost_fc=105,
         bass_boost_q=0.76,
         tilt=0.0,
         do_room_correction=True,
         do_headphone_compensation=True,
         do_equalization=True):
    """"""
    if dir_path is None or not os.path.isdir(dir_path):
        raise NotADirectoryError(f'Given dir path "{dir_path}"" is not a directory.')

    # Dir path as absolute
    dir_path = os.path.abspath(dir_path)

    # Impulse response estimator
    estimator = open_impulse_response_estimator(dir_path, file_path=test_signal)

    # Room correction frequency responses
    room_frs = None
    if do_room_correction:
        _, room_frs = room_correction(
            estimator, dir_path,
            target=room_target,
            mic_calibration=room_mic_calibration,
            fr_combination_method=fr_combination_method,
            specific_limit=specific_limit,
            generic_limit=generic_limit,
            plot=plot
        )

    # Headphone compensation frequency responses
    hp_left, hp_right = None, None
    if do_headphone_compensation:
        hp_left, hp_right = headphone_compensation(estimator, dir_path)

    # Equalization
    eq_left, eq_right = None, None
    if do_equalization:
        eq_left, eq_right = equalization(estimator, dir_path)

    # Bass boost and tilt
    target = create_target(estimator, bass_boost_gain, bass_boost_fc, bass_boost_q, tilt)

    # HRIR measurements
    hrir = open_binaural_measurements(estimator, dir_path)

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
                fr = FrequencyResponse(
                    name=f'{speaker}-{side} eq',
                    frequency=FrequencyResponse.generate_frequencies(f_step=1.01, f_min=10, f_max=estimator.fs / 2),
                    raw=0, error=0
                )

                if room_frs is not None and speaker in room_frs and side in room_frs[speaker]:
                    # Room correction
                    fr.error += room_frs[speaker][side].error

                hp_eq = hp_left if side == 'left' else hp_right
                if hp_eq is not None:
                    # Headphone compensation
                    fr.error += hp_eq.error

                eq = eq_left if side == 'left' else eq_right
                if eq is not None and type(eq) == FrequencyResponse:
                    # Equalization
                    fr.error += eq.error

                # Remove bass and tilt target from the error
                fr.error -= target.raw

                # Smoothen and equalize
                fr.smoothen_heavy_light()
                fr.equalize(max_gain=40, treble_f_lower=10000, treble_f_upper=estimator.fs / 2)

                # Create FIR filter and equalize
                # FIXME: This is 50% of execution time
                fir = fr.minimum_phase_impulse_response(fs=estimator.fs, normalize=False, f_res=5)
                ir.equalize(fir)

    # Correct channel balance
    if channel_balance is not None:
        # FIXME: This is 25% of execution time
        hrir.correct_channel_balance(channel_balance)

    # Normalize gain
    hrir.normalize(peak_target=None if target_level is not None else -0.1, avg_target=target_level)

    if plot:
        # Convolve test signal, re-plot waveform and spectrogram
        for speaker, pair in hrir.irs.items():
            for side, ir in pair.items():
                ir.recording = ir.convolve(estimator.test_signal)
        # Plot post processing
        hrir.plot(os.path.join(dir_path, 'plots', 'post'))

    # Plot results, always
    hrir.plot_result(os.path.join(dir_path, 'plots'))

    # Re-sample
    if fs is not None and fs != hrir.fs:
        hrir.resample(fs)
        hrir.normalize(peak_target=None if target_level is not None else -0.1, avg_target=target_level)

    # Write multi-channel WAV file with standard track order
    hrir.write_wav(os.path.join(dir_path, 'hrir.wav'))

    # Write multi-channel WAV file with HeSuVi track order
    hrir.write_wav(os.path.join(dir_path, 'hesuvi.wav'), track_order=HESUVI_TRACK_ORDER)


def open_impulse_response_estimator(dir_path, file_path=None):
    """Opens impulse response estimator from a file

    Args:
        dir_path: Path to directory
        file_path: Explicitly given (if any) path to impulse response estimator Pickle or test signal WAV file

    Returns:
        ImpulseResponseEstimator instance
    """
    if file_path is None:
        # Test signal not explicitly given, try Pickle first then WAV
        if os.path.isfile(os.path.join(dir_path, 'test.pkl')):
            file_path = os.path.join(dir_path, 'test.pkl')
        elif os.path.isfile(os.path.join(dir_path, 'test.wav')):
            file_path = os.path.join(dir_path, 'test.wav')
    if re.match(r'^.+\.wav$', file_path, flags=re.IGNORECASE):
        # Test signal is WAV file
        estimator = ImpulseResponseEstimator.from_wav(file_path)
    elif re.match(r'^.+\.pkl$', file_path, flags=re.IGNORECASE):
        # Test signal is Pickle file
        estimator = ImpulseResponseEstimator.from_pickle(file_path)
    else:
        raise TypeError(f'Unknown file extension for test signal "{file_path}"')
    return estimator


def equalization(estimator, dir_path):
    """Reads equalization FIR filter or CSV settings

    Args:
        estimator: ImpulseResponseEstimator
        dir_path: Path to directory

    Returns:
        - Left side FIR as Numpy array or FrequencyResponse or None
        - Right side FIR as Numpy array or FrequencyResponse or None
    """
    if os.path.isfile(os.path.join(dir_path, 'eq.wav')):
        print('eq.wav is no longer supported, use eq.csv!')
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
    if right_fr is not None and right_fr != left_fr:
        right_fr.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)

    # Plot
    if left_fr is not None or right_fr is not None:
        if left_fr == right_fr:
            # Both are the same, plot only one graph
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 9)
            left_fr.plot_graph(fig=fig, ax=ax, show=False)
        else:
            # Left and right are different, plot two graphs in the same figure
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(22, 9)
            if left_fr is not None:
                left_fr.plot_graph(fig=fig, ax=ax[0], show=False)
            if right_fr is not None:
                right_fr.plot_graph(fig=fig, ax=ax[1], show=False)
        save_fig_as_png(os.path.join(dir_path, 'plots', 'eq.png'), fig)

    return left_fr, right_fr


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
    gain_l = _left.center([100, 10000])
    gain_r = _right.center([100, 10000])
    ax = fig.add_subplot(gs[:, 1:])
    ax.plot(_left.frequency, _left.raw, linewidth=1, color='#1f77b4')
    ax.plot(_right.frequency, _right.raw, linewidth=1, color='#d62728')
    ax.plot(_left.frequency, _left.raw - _right.raw, linewidth=1, color='#680fb9')
    sl = np.logical_and(_left.frequency > 20, _left.frequency < 20000)
    stack = np.vstack([_left.raw[sl], _right.raw[sl], _left.raw[sl] - _right.raw[sl]])
    ax.set_ylim([np.min(stack) * 1.1, np.max(stack) * 1.1])
    axl.set_ylim([np.min(stack) * 1.1, np.max(stack) * 1.1])
    axr.set_ylim([np.min(stack) * 1.1, np.max(stack) * 1.1])
    ax.set_title('Comparison')
    ax.legend([f'Left raw {gain_l:+.1f} dB', f'Right raw {gain_r:+.1f} dB', 'Difference'], fontsize=8)
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


def create_target(estimator, bass_boost_gain, bass_boost_fc, bass_boost_q, tilt):
    """Creates target frequency response with bass boost, tilt and high pass at 20 Hz"""
    target = FrequencyResponse(
        name='bass_and_tilt',
        frequency=FrequencyResponse.generate_frequencies(f_min=10, f_max=estimator.fs / 2, f_step=1.01)
    )
    target.raw = target.create_target(
        bass_boost_gain=bass_boost_gain,
        bass_boost_fc=bass_boost_fc,
        bass_boost_q=bass_boost_q,
        tilt=tilt
    )
    high_pass = FrequencyResponse(
        name='high_pass',
        frequency=[10, 18, 19, 20, 21, 22, 20000],
        raw=[-80, -5, -1.6, -0.6, -0.2, 0, 0]
    )
    high_pass.interpolate(f_min=10, f_max=estimator.fs / 2, f_step=1.01)
    target.raw += high_pass.raw
    return target


def open_binaural_measurements(estimator, dir_path):
    """Opens binaural measurement WAV files.

    Args:
        estimator: ImpulseResponseEstimator
        dir_path: Path to directory

    Returns:
        HRIR instance
    """
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
    return hrir


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
    arg_parser.add_argument('--target_level', type=float, default=argparse.SUPPRESS,
                            help='Target average gain level for left and right channels. This will sum together all '
                                 'left side impulse responses and right side impulse responses respectively and take '
                                 'the average gain from mid frequencies. The averaged level is then normalized to the '
                                 'given target level. This makes it possible to compare HRIRs with somewhat similar '
                                 'loudness levels. This should be negative in most cases to avoid clipping.')
    arg_parser.add_argument('--fr_combination_method', type=str, default='average',
                            help='Method for combining frequency responses of generic room measurements if there are '
                                 'more than one tracks in the file. "average" will simply average the frequency'
                                 'responses. "conservative" will take the minimum absolute value for each frequency '
                                 'but only if the values in all the measurements are positive or negative at the same '
                                 'time.')
    arg_parser.add_argument('--specific_limit', type=float, default=10000,
                            help='Upper limit for room equalization with speaker-ear specific room measurements. '
                                 'Equalization will drop down to 0 dB at this frequency in the leading octave. 0 '
                                 'disables limit.')
    arg_parser.add_argument('--generic_limit', type=float, default=1000,
                            help='Upper limit for room equalization with generic room measurements. '
                                 'Equalization will drop down to 0 dB at this frequency in the leading octave. 0 '
                                 'disables limit.')
    arg_parser.add_argument('--bass_boost', type=str, default=argparse.SUPPRESS,
                            help='Bass boost shelf. Sub-bass frequencies will be boosted by this amount. Can be '
                                 'either a single value for a gain in dB or a comma separated list of three values for '
                                 'parameters of a low shelf filter, where the first is gain in dB, second is center '
                                 'frequency (Fc) in Hz and the last is quality (Q). When only a single value (gain) is '
                                 'given, default values for Fc and Q are used which are 105 Hz and 0.76, respectively. '
                                 'For example "--bass_boost=6" or "--bass_boost=6,150,0.69".')
    arg_parser.add_argument('--tilt', type=float, default=argparse.SUPPRESS,
                            help='Target tilt in dB/octave. Positive value (upwards slope) will result in brighter '
                                 'frequency response and negative value (downwards slope) will result in darker '
                                 'frequency response. 1 dB/octave will produce nearly 10 dB difference in '
                                 'desired value between 20 Hz and 20 kHz. Tilt is applied with bass boost and both '
                                 'will affect the bass gain.')
    args = vars(arg_parser.parse_args())
    if 'bass_boost' in args:
        bass_boost = args['bass_boost'].split(',')
        if len(bass_boost) == 1:
            args['bass_boost_gain'] = float(bass_boost[0])
            args['bass_boost_fc'] = 105
            args['bass_boost_q'] = 0.76
        elif len(bass_boost) == 3:
            args['bass_boost_gain'] = float(bass_boost[0])
            args['bass_boost_fc'] = float(bass_boost[1])
            args['bass_boost_q'] = float(bass_boost[2])
        else:
            raise ValueError('"--bass_boost" must have one value or three values separated by commas!')
        del args['bass_boost']
    return args


if __name__ == '__main__':
    main(**create_cli())
