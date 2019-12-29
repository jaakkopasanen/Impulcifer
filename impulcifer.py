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
from PIL import Image
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
         plot=False,
         channel_balance=None,
         do_room_correction=True,
         do_headphone_compensation=True):
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

    # Compensate headphones
    headphone_firs = None
    if os.path.isfile(headphones) and do_headphone_compensation:
        headphone_firs = headphone_compensation(headphones, estimator, dir_path=dir_path)

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

    # Room correction
    if do_room_correction:
        correct_room(
            hrir,
            dir_path=dir_path,
            room_target=room_target,
            room_mic_calibration=room_mic_calibration,
            plot=plot
        )

    if headphone_firs is not None:
        hrir.equalize(headphone_firs)

    # Apply given equalization filter
    if os.path.isfile(eq):
        eq_fs, firs = read_wav(eq)
        if eq_fs != hrir.fs:
            raise ValueError('Equalization FIR filter sampling rate must match HRIR sampling rate.')
        hrir.equalize(firs)

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
        room_target.interpolate(f_step=1.05, f_min=10, f_max=hrir.fs / 2)
        room_target.center()
    else:
        # No room target specified, use flat
        room_target = FrequencyResponse(name='room-target')
        room_target.raw = np.zeros(room_target.frequency.shape)
        room_target.interpolate(f_step=1.05, f_min=10, f_max=hrir.fs / 2)

    # Room measurement microphone calibration
    if room_mic_calibration is None:
        room_mic_calibration = os.path.join(dir_path, 'room-mic-calibration.csv')
        if not os.path.isfile(room_mic_calibration):
            room_mic_calibration = os.path.join(dir_path, 'room-mic-calibration.txt')
    elif not os.path.isfile(room_mic_calibration):
        raise FileNotFoundError(f'Room mic calibration file doesn\'t exist at "{room_mic_calibration}"')
    if os.path.isfile(room_mic_calibration):
        # File found, create frequency response
        room_mic_calibration = FrequencyResponse.read_from_csv(room_mic_calibration)
        room_mic_calibration.interpolate(f_step=1.05, f_min=10, f_max=hrir.fs / 2)
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
    fr_axes = []
    for speaker, pair in rir.irs.items():
        for side, ir in pair.items():
            fr = ir.frequency_response()
            fr.interpolate(f_step=1.05, f_min=10, f_max=hrir.fs / 2)

            if room_mic_calibration is not None:
                # Calibrate frequency response
                fr.raw -= room_mic_calibration.raw

            # Save original data for gain syncing
            original = fr.raw.copy()

            # Process
            fr.center()
            fr.compensate(room_target, min_mean_error=True)
            fr.smoothen_heavy_light()
            fr.equalize(
                max_gain=30,
                smoothen=True,
                treble_f_lower=10000,
                treble_f_upper=20000
            )

            # Sync gains
            sl = np.logical_and(fr.frequency >= 100, fr.frequency <= 10000)
            gain = np.mean(original[sl] + fr.equalization[sl])
            if ref_gain is None:
                ref_gain = gain
            fir_gain = ref_gain - gain
            fir_gain += IR_ROOM_SPL[speaker][side]
            fir_gain = 10**(fir_gain / 20)

            # TODO: Some alien-tech mixed phase filter
            fir = fr.minimum_phase_impulse_response(fs=rir.fs, f_res=4, normalize=False)
            fir *= fir_gain
            # Equalize
            hrir.irs[speaker][side].equalize(fir)

            if plot:
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                _, fr_ax = ir.plot_fr(
                    fr=fr,
                    fig=figs[speaker][side],
                    ax=figs[speaker][side].get_axes()[4],
                    plot_raw=True,
                    plot_error=False,
                    plot_file_path=file_path
                )
                fr_axes.append(fr_ax)

    # Zero pad all IRs to same length
    # TODO: Check delay sync with mixed phase filter
    lengths = []
    for speaker, pair in hrir.irs.items():
        for side, ir in pair.items():
            lengths.append(len(ir))
    n = max(lengths)
    for speaker, pair in hrir.irs.items():
        for side, ir in pair.items():
            pad = n - len(ir)
            if pad > 0:
                ir.data = np.concatenate([ir.data, np.zeros((pad,))])

    if plot:
        # Sync FR plot axes
        sync_axes(fr_axes)
        # Save figures
        for speaker, pair in figs.items():
            for side, fig in pair.items():
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                fig.savefig(file_path, bbox_inches='tight')
                plt.close(fig)
                # Optimize file size
                im = Image.open(file_path)
                im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
                im.save(file_path, optimize=True)

    return rir


def headphone_compensation(recording, estimator, dir_path=None):
    """Equalizes HRIR tracks with headphone compensation measurement.

    Args:
        recording: File path to sine sweep recording made with headphones
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to output directory

    Returns:
        None
    """
    # Read WAV file
    hp_irs = HRIR(estimator)
    hp_irs.open_recording(recording, speakers=['FL', 'FR'])
    hp_irs.write_wav(os.path.join(dir_path, 'headphone-responses.wav'))
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
        eq_ir = fr.minimum_phase_impulse_response(fs=estimator.fs, f_res=4)
        firs.append(ImpulseResponse(eq_ir, estimator.fs))
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

    if dir_path is not None:
        # Headphone plots
        fig = plt.figure()
        gs = fig.add_gridspec(2, 3)
        fig.set_size_inches(22, 10)
        fig.suptitle('Headphones')

        # Left
        axl = fig.add_subplot(gs[0, 0])
        frs[0].plot_graph(fig=fig, ax=axl, show=False)
        axl.set_title('Left')
        # Right
        axr = fig.add_subplot(gs[1, 0])
        frs[1].plot_graph(fig=fig, ax=axr, show=False)
        axr.set_title('Right')
        # Sync axes
        sync_axes([axl, axr])

        # Combined
        frs[0].center([100, 10000])
        frs[1].center([100, 10000])
        ax = fig.add_subplot(gs[:, 1:])
        ax.plot(frs[0].frequency, frs[0].raw, linewidth=1, color='#1f77b4')
        ax.plot(frs[1].frequency, frs[1].raw, linewidth=1, color='#d62728')
        ax.plot(frs[0].frequency, frs[0].raw - frs[1].raw, linewidth=1, color='#680fb9')
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
        os.makedirs(os.path.join(dir_path, 'plots'), exist_ok=True)
        file_path = os.path.join(dir_path, 'plots', 'Headphones.png')
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
        # Optimize file size
        im = Image.open(file_path)
        im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
        im.save(file_path, optimize=True)

    return firs


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
