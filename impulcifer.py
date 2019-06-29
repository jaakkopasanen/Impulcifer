# -*- coding: utf-8 -*-

import os
from time import time
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import argparse
from scipy import signal
from scipy.signal import kaiser, hanning, convolve
from autoeq.frequency_response import FrequencyResponse
from impulse_response_estimator import ImpulseResponseEstimator
from hrir import HRIR
from impulse_response import ImpulseResponse
from utils import read_wav, write_wav, magnitude_response, sync_axes
from constants import IR_ORDER, SPEAKER_DELAYS


def to_float(x):
    """Normalizes numpy array into range -1..1

    Args:
        x: Numpy array

    Returns:
        Numpy array with values in range -1..1
    """
    if type(x) != np.array:
        x = np.array(x)
    dtype = x.dtype
    x = x.astype('float64')
    if dtype == 'int32':
        x /= 2.0 ** 31
    elif dtype == 'int16':
        x /= 2.0 ** 15
    elif dtype == 'uint8':
        x /= 2.0 ** 8
        x *= 2.0
        x -= 1.0
    return x


def plot_sweep(x, fs, name=None):
    plt.plot(np.arange(0, len(x) / fs, 1 / fs), x)
    plt.xlim([0.0, len(x) / fs])
    plt.ylim([-1.0, 1.0])
    plt.ylabel('Amplitude')
    if name is not None:
        plt.legend([name])


def split_recording(recording, test_signal, speakers, fs, silence_length):
    """Splits sine sweep recording into individual speaker-ear pairs

    Recording looks something like this (stereo only in this example):
    --/\/\/\----/\/\/\--
    ---/\/\/\--/\/\/\---
    There are two tracks, one for each ear. Dashes represent silence and sawtooths recorded signal. First saw tooths
    of both tracks are signal played on left front speaker and the second ones are signal played of right front speaker.

    There can be any (even) number of tracks. Output will be two tracks per speaker, left ear first and then right.
    Speakers are in the same order as in the original file, read from left to right and top to bottom. In the example
    above the output track order would be:
    1. Front left speaker - left ear
    2. Front left speaker - right ear
    3. Front right speaker - left ear
    4. Front right speaker - right ear

    Args:
        recording: Numpy array for the sine sweep recording
        test_signal: Numpy array for the test signal
        speakers: Speaker order as a list of strings
        fs: Sampling rate
        silence_length: Length of silence in the beginning, end and between test signals in seconds

    Returns:

    """
    if silence_length * fs != int(silence_length * fs):
        raise ValueError('Silence length must produce full samples with given sampling rate.')
    silence_length = int(silence_length * fs)

    # Number of speakers in each track
    n_columns = round(len(speakers) / (recording.shape[0] // 2))

    # Crop out initial silence
    recording = recording[:, silence_length:]

    # Split sections in time to columns
    columns = []
    column_size = silence_length + len(test_signal)
    for i in range(n_columns):
        columns.append(recording[:, i*column_size:(i+1)*column_size])

    # Split each track by columns
    tracks = []
    i = 0
    while i < recording.shape[0]:
        for column in columns:
            tracks.append(column[i, :])  # Left ear of current speaker
            tracks.append(column[i+1, :])  # Right ear of current speaker
        i += 2
    tracks = np.vstack(tracks)

    return tracks


def normalize(x, target_db=-0.1):
    arg_max = np.unravel_index(np.argmax(x), x.shape)  # 2D argmax index
    x = x / np.abs(x[arg_max])  # Normalize by largest abs value
    x = x * 10**(target_db / 20)  # Scale down by -0.1 dB
    return x


def impulse_response_decay(impulse_response, fs, window_size_ms=1, fig=None, ax=None, show_plot=False, plot_file_path=None):
    # Sliding window RMS
    window_size = fs // 1000 * window_size_ms

    # RMS windows
    n = len(impulse_response) // window_size
    windows = np.vstack(np.split(impulse_response[:n * window_size], n))
    rms = np.sqrt(np.mean(np.square(windows), axis=1))

    rms[rms == 0.0] = 0.001

    # Smoothen data
    smoothed = 20*np.log10(rms)
    for _ in range(200):
        smoothed = signal.savgol_filter(smoothed, 11, 1)

    if show_plot or plot_file_path or fig:
        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(window_size_ms * np.arange(len(rms)), 20*np.log10(rms), linewidth=0.5)
        ax.set_ylim([-150, 0])
        ax.set_xlim([-100, len(rms) * window_size_ms])
        ax.set_xlabel('Time (ms)')
        ax.grid(True, which='major')
        ax.set_title('Decay')
        if plot_file_path:
            fig.savefig(plot_file_path)
        if show_plot:
            plt.show(fig)

    return smoothed


def tail_index(rms, rms_window_size):
    """Finds index in an impulse response after which there is nothing but noise left

    Args:
        rms: RMS values in dB for windows as numpy array
        rms_window_size: Number of samples in each RMS window

    Returns:
        Tail index
    """
    # Peak
    peak_index = int(np.argmax(rms))

    for i in range(peak_index+1, len(rms)):
        if rms[i] > rms[i-1]:
            break
    noise_floor = rms[i]
    noise_floor += 3  # +3dB headroom for algorithmic safety (this is a very simple algorith)
    tail_ind = np.argmax(rms < noise_floor)  # argmax will select first matching index because all of them are boolean

    # Tail index in impulse response, rms has larger window size
    tail_ind *= rms_window_size

    return int(tail_ind)


def crop_ir_head(left, right, speaker, fs, head_ms=1):
    """Crops out silent head of left and right ear impulse responses and sets delay to correct value according to
    speaker channel

    Args:
        left: Left ear impulse response
        right: Right ear impulse response
        speaker: Speaker channel name
        fs: Sampling rate

    Returns:
        Cropped left and right AudioSegments as a tuple (left, right)
    """
    # Peaks
    peak_left, _ = signal.find_peaks(left / np.max(left), height=0.1)
    peak_left = peak_left[0]
    peak_right, _ = signal.find_peaks(right / np.max(right), height=0.1)
    peak_right = peak_right[0]
    # Inter aural time difference (in samples)
    itd = np.abs(peak_left - peak_right)

    # Speaker channel delay
    head = head_ms * fs // 1000
    delay = int(np.round(SPEAKER_DELAYS[speaker] / 1000 * fs)) + head  # Channel delay in samples

    if peak_left < peak_right:
        # Delay to left ear is smaller, this is must left side speaker
        if speaker[1] == 'R':
            # Speaker name indicates this is right side speaker, there is something wrong with the measurement
            raise ValueError(speaker + ' impulse response has lower delay to left ear than to right.')
        # Crop out silence from the beginning, only required channel delay remains
        # Secondary ear has additional delay for inter aural time difference
        left = left[peak_left-delay:]
        right = right[peak_right-(delay+itd):]
    else:
        # Delay to right ear is smaller, this is must right side speaker
        if speaker[1] == 'L':
            # Speaker name indicates this is left side speaker, there si something wrong with the measurement
            raise ValueError(speaker + ' impulse response has lower delay to right ear than to left.')
        # Crop out silence from the beginning, only required channel delay remains
        # Secondary ear has additional delay for inter aural time difference
        left = left[peak_left-(delay+itd):]
        right = right[peak_right-delay:]

    # Make sure impulse response starts from silence
    window = hanning(head*2)[:head]
    left[:head] *= window
    right[:head] *= window
    # left[0] *= 0.0
    # left[1] *= 0.5
    # right[0] *= 0.0
    # right[1] *= 0.5

    return left, right


def zero_pad(data, max_samples):
    """Zero pads data to a give length

    Args:
        data: Audio data as numpy array
        max_samples: Target number of samples

    Returns:

    """
    padding_length = max_samples - data.shape[1]
    silence = np.zeros((data.shape[0], padding_length), dtype=data.dtype)
    zero_padded = np.concatenate([data, silence], axis=1)
    return zero_padded


def reorder_tracks(tracks, speakers):
    """Reorders tracks to match standard track order

    Will add silent tracks if the given tracks do not contain all seven speakers

    Args:
        tracks: Sample data for tracks as Numpy array, tracks on rows
        speakers: List of speaker names eg. ["FL", "FR"]

    Returns:
        Reordered tracks
    """
    track_names = []
    for speaker in speakers:
        track_names.append(speaker + '-left')
        track_names.append(speaker + '-right')

    reordered = []
    for ch in IR_ORDER:
        if ch not in track_names:
            reordered.append(np.zeros(tracks.shape[1]))
        else:
            reordered.append(tracks[track_names.index(ch)])
    reordered = np.vstack(reordered)
    return reordered


def spectrogram(sweep, fs, fig=None, ax=None, show_plot=False, plot_file_path=None):
    """Plots spectrogram for a logarithmic sine sweep recording.

    Args:
        sweep: Recording data
        fs: Sampling rate
        fig: Figure
        ax: Axis
        show_plot: Show plot live?
        plot_file_path: Path to a file for saving the plot

    Returns:
        None
    """
    if len(np.nonzero(sweep)[0]) == 0:
        return

    if fig is None:
        fig, ax = plt.subplots()
    ax.specgram(sweep, Fs=fs)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')

    if plot_file_path:
        fig.savefig(plot_file_path)
    if show_plot:
        plt.show(fig)


def plot_ir(ir, fs, fig=None, ax=None, max_time=None, show_plot=False, plot_file_path=None):
    """Plots impulse response wave form.

    Args:
        ir: Impulse response data
        fs: Sampling rate
        max_time: Maximum time in seconds for cropping the tail.
        show_plot: Show plot live?
        plot_file_path: Path to a file for saving the plot

    Returns:
        None
    """
    if len(np.nonzero(ir)[0]) == 0:
        return

    if max_time is None:
        max_time = len(ir) / fs
    ir = ir[:int(max_time * fs)]

    if fig is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(ir)/fs*1000, 1000/fs), ir, linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)
    ax.set_title('Impulse response {ms} ms'.format(ms=int(max_time*1000)))

    if plot_file_path:
        fig.savefig(plot_file_path)
    if show_plot:
        plt.show()


def main(dir_path=None,
         test_signal=None,
         speakers=None,
         compensate_headphones=False):
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
    hrir.open_recording(recording, speakers=speakers)

    # Write multi-channel WAV file with sine sweeps for debugging
    hrir.write_wav(os.path.join(dir_path, 'responses.wav'))

    # Plot
    hrir.plot(dir_path=dir_path)
    # plt.show()

    # Crop noise and harmonics from the beginning
    hrir.crop_heads()

    # Crop noise from the tail
    hrir.crop_tails()

    if compensate_headphones:
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

        # Headphone plots
        plots = {'left': {'fr': None, 'ir': None}, 'right': {'fr': None, 'ir': None}}
        for ir, fr, side in zip(eq_irs, frs, ['left', 'right']):
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(15, 7)
            fr.plot_graph(fig=fig, ax=ax[0], show=False)
            ir.plot_ir(fig=fig, ax=ax[1], max_time=2e-3)
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
            fig.savefig(os.path.join(dir_path, f'Headphones {side}.png'))
            plt.close(fig)

        # Equalize HRIR with headphone compensation FIR filters
        for speaker, pair in hrir.irs.items():
            for side, ir in pair.items():
                if side == 'left':
                    hrir.irs[speaker][side].data = convolve(ir.data, eq_irs[0].data, mode='full')
                else:
                    hrir.irs[speaker][side].data = convolve(ir.data, eq_irs[1].data, mode='full')

    # Normalize gain
    hrir.normalize(target_db=12)

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
    arg_parser.add_argument('--compensate_headphones', action='store_true',
                            help='Produce CSV file for AutoEQ from headphones sine sweep recordgin?')
    arg_parser.add_argument('--dir_path', type=str, help='Path to directory for recordings and outputs.')
    arg_parser.add_argument('--test_signal', type=str, help='File path to sine sweep test signal.')
    arg_parser.add_argument('--speakers', type=str,
                            help='Order of speakers in the recording as a comma separated list of speaker channel '
                                 'names. Supported names are "FL" (front left), "FR" (front right), '
                                 '"FC" (front center), "BL" (back left), "BR" (back right), '
                                 '"SL" (side left), "SR" (side right)". For example: "FL,FR".')
    args = vars(arg_parser.parse_args())
    if 'speakers' in args and args['speakers'] is not None:
        args['speakers'] = args['speakers'].upper().split(',')
    return args


if __name__ == '__main__':
    main(**create_cli())
