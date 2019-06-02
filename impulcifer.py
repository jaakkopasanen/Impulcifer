# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pydub import AudioSegment
import argparse
from scipy import signal
from scipy.signal import fftconvolve, kaiser, convolve
import pyfftw
from time import time
from PIL import Image
from impulse_response_estimator import ImpulseResponseEstimator
from autoeq.frequency_response import FrequencyResponse

# https://en.wikipedia.org/wiki/Surround_sound
SPEAKER_NAMES = ['FL', 'FR', 'FC', 'BL', 'BR', 'SL', 'SR']

# Each channel, left and right
IR_ORDER = []
for _ch in SPEAKER_NAMES:
    IR_ORDER.append(_ch+'-left')
    IR_ORDER.append(_ch+'-right')

# Time delays between speaker to primary ear vs speaker to middle of head in milliseconds
# Speaker configuration is perfect circle around listening position
# Distance from side left speaker (SL) to left ear is smaller than distance from front left (FL) to left ear
# Two samples (@48kHz) added to each for head room
# These are used for synchronizing impulse responses
# TODO: Remove the two samples from these delays and add the headroom in the code
SPEAKER_DELAYS = {
    'FL': 0.107,
    'FR': 0.107,
    'FC': 0.214,
    'BL': 0.107,  # TODO: Confirm this
    'BR': 0.107,  # TODO: Confirm this
    'SL': 0.0,
    'SR': 0.0,
}


def magnitude_response(x, fs):
    """Calculates frequency magnitude response

    Args:
        x: Audio data
        fs: Sampling rate

    Returns:
        - **f:** Frequencies
        - **X:** Magnitudes
    """
    nfft = len(x)
    df = fs / nfft
    f = np.arange(0, fs - df, df)
    X = np.fft.fft(x)
    X_mag = 20 * np.log10(np.abs(X))
    return f[0:int(np.ceil(nfft/2))], X_mag[0:int(np.ceil(nfft/2))]


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
        recording: AudioSegment for the sine sweep recording
        test_signal: AudioSegment for the test signal
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
    column_size = silence_length + test_signal.shape[1]
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
            fig.show()

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
    # TODO: This should by Kaiser (or other window function) and have more samples
    window = kaiser(head*2, 16)[:head]
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


def deconv(recording, test_signal, method='inverse_filter', fs=None):
    """Calculates deconvolution in frequency or time domain.

    Args:
        recording: Recording as numpy array
        test_signal: Test signal as numpy array
        method: "inverse_filter" or "fft"
        fs: Sampling rate, required when method is "inverse_filter"

    Returns:
        Impulse response
    """
    if method == 'fft':
        # Division in frequency domain is deconvolution in time domain
        X = np.fft.fft(test_signal)
        Y = np.fft.fft(recording)
        H = Y / X
        h = np.fft.ifft(H)
        h = np.real(h)

    elif method == 'inverse_filter':
        if fs is None:
            raise TypeError('Sampling rate is required for inverse filter deconvolution.')
        t = time()
        test_signal = np.squeeze(test_signal)
        # TODO: Use ImpulseResponseEstimator to avoid building inverse filter for every track

        # FIXME: low and high should be given as parameters, it's not possible to infer them from test signal
        low = 20
        high = 20000
        w1 = low / fs * 2*np.pi  # Sweep start frequency in radians relative to sampling frequency
        w2 = high / fs * 2*np.pi  # Sweep end frequency in radians relative to sampling frequency

        # This is what the value of K will be at the end (in dB):
        k_end = 10 ** ((-6 * np.log2(w2 / w1)) / 20)
        # dB to rational number.
        k = np.log(k_end) / len(test_signal)

        # Making inverse of test signal so that convolution will just calculate a dot product
        # Weighting it with exponent to achieve 6 dB per octave amplitude decrease.
        c = np.array(list(map(lambda t: np.exp(float(t) * k), range(len(test_signal)))))
        inv_filter = np.flip(test_signal) * c

        # Now we have to normalize energy of result of dot product.
        # This is "naive" method but it just works.
        frp = pyfftw.empty_aligned(len(test_signal)*2-1, dtype='complex128')
        frp[:] = fftconvolve(inv_filter, test_signal)
        frp = pyfftw.interfaces.scipy_fftpack.fft(frp)
        inv_filter /= np.abs(frp[round(frp.shape[0] / 4)])

        # Deconvolution between recording and inverse filter
        h = fftconvolve(recording, inv_filter, mode='full')
        h = np.concatenate([h, [0.0]])
        h = h[len(test_signal):len(test_signal) * 2 + 1]
    else:
        raise ValueError('"{}" is not one of the supported "domain" parameter values "time" or "frequency".')
    return h


def read_wav(file_path):
    """Reads WAV file

    Args:
        file_path: Path to WAV file as string

    Returns:
        - sampling frequency as integer
        - wav data as numpy array with one row per track, samples in range -1..1
    """
    # Using AudioSegment because SciPy can't read 24-bit WAVs
    seg = AudioSegment.from_wav(file_path)
    data = []
    for track in seg.split_to_mono():
        # Read samples of each track separately
        data.append(track.get_array_of_samples())
    # Create numpy array where tracks are on rows and samples have been scaled in range -1..1
    data = to_float(np.vstack(data))
    return seg.frame_rate, data


def write_wav(file_path, fs, data):
    """Writes WAV file."""
    tracks = []
    if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
    for i in range(data.shape[0]):
        tracks.append(AudioSegment(
            np.multiply(data[i, :], 2 ** 31).astype('int32').tobytes(),
            frame_rate=fs,
            sample_width=4,
            channels=1
        ))
    seg = AudioSegment.from_mono_audiosegments(*tracks)
    seg.export(file_path, format='wav')


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
    plt.specgram(sweep, Fs=fs)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')

    if plot_file_path:
        fig.savefig(plot_file_path)
    if show_plot:
        fig.show()


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


def main(measure=False,
         compensate_headphones=False,
         dir_path=None,
         recording=None,
         headphones=None,
         test_signal=None,
         speakers=None,
         silence_length=None):
    """"""
    out_dir = 'out'
    if dir_path and os.path.isdir(dir_path):
        out_dir = dir_path
        if not recording and os.path.isfile(os.path.join(dir_path, 'recording.wav')):
            recording = os.path.join(dir_path, 'recording.wav')
        if not test_signal and os.path.isfile(os.path.join(dir_path, 'test.wav')):
            test_signal = os.path.join(dir_path, 'test.wav')
        if not headphones and os.path.isfile(os.path.join(dir_path, 'headphones.wav')):
            headphones = os.path.join(dir_path, 'headphones.wav')

    # Read files
    fs_rec, recording = read_wav(recording)
    fs_ts, test_signal = read_wav(test_signal)
    if fs_rec != fs_ts:
        raise ValueError('Sampling rates of recording and test signal do not match.')
    fs = fs_rec

    if not os.path.isdir(out_dir):
        # Output directory does not exist, create it
        os.makedirs(out_dir, exist_ok=True)

    plots = dict()
    for ch_name in IR_ORDER:
        fig, ax = plt.subplots(2, 2, num=ch_name)
        plt.suptitle(ch_name)
        fig.set_size_inches(15, 10)
        plots[ch_name] = {
            'figure': fig,
            'ir': ax[0, 0],
            'fr': ax[0, 1],
            'decay': ax[1, 0],
            'spectrogram': ax[1, 1],
        }

    # Logarithmic sine sweep measurement
    if measure:  # TODO
        raise NotImplementedError('Measurement is not yet implemented.')

    # Split recording WAV file into individual mono tracks
    recording = split_recording(recording, test_signal, speakers, fs, silence_length)

    # Reorder tracks to match standard
    recording = reorder_tracks(recording, speakers)

    # Normalize to -0.1 dB
    recording = normalize(recording, target_db=-0.1)

    # for i in range(recording.shape[0]):
    #     spectrogram(
    #         recording[i, :],
    #         fs,
    #         fig=plots[IR_ORDER[i]]['figure'],
    #         ax=plots[IR_ORDER[i]]['spectrogram']
    #     )

    # Write multi-channel WAV file with sine sweeps for debugging
    write_wav(os.path.join(out_dir, 'preprocessed.wav'), fs, recording)

    # Estimate impulse responses by deconvolution
    # TODO: duration and low should be parameters
    estimator = ImpulseResponseEstimator(duration=5, low=10, fs=fs)
    impulse_responses = []
    for i in range(recording.shape[0]):
        track = recording[i, :]
        if len(np.nonzero(track)[0]) > 0:
            # Run deconvolution
            impulse_response = estimator.estimate(track)
            # Add to responses
            impulse_responses.append(impulse_response)
        else:
            # Silent track
            impulse_responses.append(np.zeros(estimator.test_signal.shape[0] + 1))
    impulse_responses = np.vstack(impulse_responses)

    # Save impulse responses to file for debugging
    write_wav(os.path.join(out_dir, 'responses.wav'), fs, impulse_responses)

    # Crop heads
    cropped = []
    i = 0
    while i < impulse_responses.shape[0]:
        # Speaker tracks are paired so that first is left ear mic and second is right ear mic
        left = impulse_responses[i]
        right = impulse_responses[i + 1]
        speaker = SPEAKER_NAMES[i // 2]
        if len(np.nonzero(left)[0]) > 0 and len(np.nonzero(right)[0]) > 0:
            impulse_response_decay(
                left,
                fs,
                fig=plots[IR_ORDER[i]]['figure'],
                ax=plots[IR_ORDER[i]]['decay'],
                window_size_ms=1,
                show_plot=False,
            )
            impulse_response_decay(
                right,
                fs,
                fig=plots[IR_ORDER[i+1]]['figure'],
                ax=plots[IR_ORDER[i+1]]['decay'],
                window_size_ms=1,
                show_plot=False,
            )
            # Crop head
            left, right = crop_ir_head(left, right, speaker, fs)
            cropped.append(left)
            cropped.append(right)
        elif len(np.nonzero(left)[0]) == 0 and len(np.nonzero(right)[0]) == 0:
            # Silent tracks
            cropped.append(left)
            cropped.append(right)
        else:
            raise ValueError('Left and right ear recording pair must be non-zero for both or neither.')
        i += 2

    # Crop tails together
    # Find indices after which there is only noise in each track
    tail_indices = []
    for i, track in enumerate(cropped):
        if len(np.nonzero(track)[0]) > 0:
            rms = impulse_response_decay(
                track,
                fs,
                window_size_ms=1,
            )
            tail_indices.append(tail_index(rms, rms_window_size=fs / 1000))
    # Crop all tracks by last tail index
    tail_ind = max(tail_indices)
    for i in range(len(cropped)):
        cropped[i] = cropped[i][:tail_ind]
    impulse_responses = np.vstack(cropped)

    # Set decay plot X-max to 1000ms after IR tail crop point
    for ch, obj in plots.items():
        obj['decay'].set_xlim(obj['decay'].get_xlim()[0], 1000 * tail_ind // fs + 1000)

    # Save IR waveform and frequency response plots
    for i, ir in enumerate(cropped):
        if len(np.nonzero(ir)[0]) > 0:
            plot_ir(
                ir,
                fs,
                fig=plots[IR_ORDER[i]]['figure'],
                ax=plots[IR_ORDER[i]]['ir'],
                max_time=0.1,
                show_plot=False,
            )
            f, m = magnitude_response(ir, fs)
            fr = FrequencyResponse(name='Frequency response', frequency=f[1:], raw=m[1:])
            fr.interpolate()
            fr.smoothen(
                window_size=1 / 3,
                iterations=1,
                treble_window_size=1 / 6,
                treble_iterations=1,
                treble_f_lower=100,
                treble_f_upper=1000,

            )
            ax = plots[IR_ORDER[i]]['fr']
            ax.set_xlabel('Frequency (Hz)')
            ax.semilogx()
            ax.set_xlim([20, 20e3])
            ax.set_ylabel('Amplitude (dBr)')
            ax.set_title(fr.name)
            ax.grid(True, which='major')
            ax.grid(True, which='minor')
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
            ax.plot(fr.frequency, fr.raw, linewidth=0.5)
            ax.plot(fr.frequency, fr.smoothed, linewidth=1)
            ax.legend(['Raw', 'Smoothed'], fontsize=8)

    # Sync axes
    ir_ylim = [0.0, 0.0]
    fr_ylim = [0.0, 0.0]
    for ch, obj in plots.items():
        if not obj['ir'].lines or not obj['fr'].lines:
            continue
        if obj['ir'].get_ylim()[0] < ir_ylim[0]:
            ir_ylim[0] = obj['ir'].get_ylim()[0]
        if obj['ir'].get_ylim()[1] > ir_ylim[1]:
            ir_ylim[1] = obj['ir'].get_ylim()[1]
        if obj['fr'].get_ylim()[0] < fr_ylim[0]:
            fr_ylim[0] = obj['fr'].get_ylim()[0]
        if obj['fr'].get_ylim()[1] > fr_ylim[1]:
            fr_ylim[1] = obj['fr'].get_ylim()[1]
    for ch, obj in plots.items():
        obj['ir'].set_ylim(ir_ylim)
        obj['fr'].set_ylim(fr_ylim)

    # Save plots
    for ch, obj in plots.items():
        if obj['ir'].lines or obj['fr'].lines or obj['spectrogram'].lines or obj['decay'].lines:
            file_path = os.path.join(out_dir, ch + '.png')
            obj['figure'].savefig(file_path, dpi=480, bbox_inches='tight')
            im = Image.open(file_path)
            im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
            im.save(file_path, optimize=True)
    plt.close('all')

    if compensate_headphones:
        # Read WAV file
        fs_hp, hp_rec = read_wav(headphones)
        # Headphones are measured one side at a time in a single sequence, split recording into tracks
        hp_rec = split_recording(hp_rec, test_signal, ['FL', 'FR'], fs_hp, silence_length)
        # Select 1st and 4th tracks which are left speaker - left microphone and right speaker - right microphone
        hp_rec = hp_rec[[0, 3], :]
        compensated = []
        eq_irs = []
        biases = []
        frs = []

        for i in range(hp_rec.shape[0]):
            # Select track
            track = hp_rec[i, :]
            # Estimate impulse response
            impulse_response = estimator.estimate(track)
            # Calculate magnitude response
            f, m = magnitude_response(impulse_response, fs_hp)
            # Create frequency response
            name = 'Left' if i == 0 else 'Right'
            fr = FrequencyResponse(name=name, frequency=f[1:], raw=m[1:])
            fr.interpolate()
            # Take copy
            fr_eq = FrequencyResponse(name='{}_eq'.format(name), frequency=fr.frequency.copy(), raw=fr.raw.copy())
            # Centering removes bias and we want the bias to stay in the original
            fr_eq.center()
            # Aiming for flat response at the ear canal
            fr_eq.compensate(
                FrequencyResponse(name='zero', frequency=fr.frequency, raw=np.zeros(len(fr.frequency))),
                min_mean_error=False
            )
            # Smoothen
            fr_eq.smoothen(
                window_size=1 / 6,
                iterations=1,
                treble_f_lower=2000,
                treble_f_upper=5000,
                treble_window_size=1 / 3,
                treble_iterations=1000
            )
            heavy = fr_eq.smoothed.copy()
            heavy_error = fr_eq.error_smoothed.copy()
            fr_eq.smoothen(
                window_size=1 / 6,
                iterations=1,
                treble_f_lower=2000,
                treble_f_upper=5000,
                treble_window_size=1 / 3,
                treble_iterations=1
            )
            light = fr_eq.smoothed.copy()
            light_error = fr_eq.error_smoothed.copy()
            combination = np.max(np.vstack([light, heavy]), axis=0)
            combination_error = np.max(np.vstack([light_error, heavy_error]), axis=0)
            sm = FrequencyResponse(name='', frequency=fr_eq.frequency.copy(), raw=combination, error=combination_error)
            sm.smoothen(
                window_size=1 / 6,
                iterations=10,
                treble_window_size=1 / 6,
                treble_iterations=10
            )
            fr_eq.smoothed = sm.smoothed.copy()
            fr_eq.error_smoothed = sm.error_smoothed.copy()
            # Equalize to flat
            fr_eq.equalize(max_gain=15, treble_f_lower=5000, treble_f_upper=20000, treble_gain_k=1)
            # Copy equalization curve
            fr.equalization = fr_eq.equalization.copy()
            # Calculate equalized raw data
            fr.equalized_raw = fr.raw + fr.equalization
            frs.append(fr)
            # Create miniumum phase FIR filter
            eq_ir = fr.minimum_phase_impulse_response(fs=fs, f_res=10)
            eq_irs.append(eq_ir)
            # Calculate bias
            avg = np.mean(fr.equalized_raw[np.logical_and(fr.frequency >= 100, fr.frequency <= 10000)])
            biases.append(avg)

        # Balance equalization filters for left and right ear
        # Both ears need to have same level
        # Levels might be different due to measurement devices or microphone placement
        if biases[0] > biases[1]:
            # Left headphone measurement is louder, bring it down to level of right headphone
            eq_irs[0] *= 10**((biases[1]-biases[0]) / 20)
            frs[0].equalization -= biases[0] - biases[1]
        else:
            # Right headphone measurement is louder, bring it down to level of left headphone
            eq_irs[1] *= 10**((biases[0] - biases[1]) / 20)
            frs[1].equalization -= biases[1] - biases[0]

        fig_l, ax_l = plt.subplots(1, 2)
        fig_l.set_size_inches(15, 7)
        plt.suptitle('Headphones Left')
        fig_r, ax_r = plt.subplots(1, 2)
        fig_r.set_size_inches(15, 7)
        plt.suptitle('Headphones Right')

        # Create FIR filter waveform plots
        for i, eq_ir in enumerate(eq_irs):
            fig = fig_l if i == 0 else fig_r
            ax = ax_l if i == 0 else ax_r
            plot_ir(eq_ir, fs, fig=fig, ax=ax[1], max_time=0.01)

        # Create headphone equalization graphs
        for i, fr in enumerate(frs):
            fr.equalized_raw = fr.raw + fr.equalization
            ax = ax_l if i == 0 else ax_r
            ax = ax[0]
            ax.plot(fr.frequency, fr.raw, color='black')
            ax.plot(fr.frequency, fr.equalization, color='darkgreen')
            ax.plot(fr.frequency, fr.equalized_raw, color='magenta')
            ax.set_xlabel('Frequency (Hz)')
            ax.semilogx()
            ax.set_xlim([20, 20e3])
            ax.set_ylim([-40, 20])
            ax.set_ylabel('Amplitude (dBr)')
            ax.set_title(fr.name)
            ax.grid(True, which='major')
            ax.grid(True, which='minor')
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
            ax.legend(['Raw', 'Equalization', 'Equalized Raw'], fontsize=8)

        # Sync FIR filter axes
        y_lim = [
            np.min([ax_l[1].get_ylim()[0], ax_r[1].get_ylim()[0]]),
            np.max([ax_l[1].get_ylim()[1], ax_r[1].get_ylim()[1]])
        ]
        ax_l[1].set_ylim(y_lim)
        ax_r[1].set_ylim(y_lim)

        # Save headphone plots
        fig_l.savefig(os.path.join(out_dir, 'Headphones Left.png'))
        plt.close(fig_l)
        fig_r.savefig(os.path.join(out_dir, 'Headphones Right.png'))
        plt.close(fig_r)

        for j in range(impulse_responses.shape[0]):
            # Equalize all left ear impulse responses with left ear headphone FIR filter and right ear impulse
            # responses with right ear headphone FIR filter
            filtered = convolve(impulse_responses[j, :], eq_irs[j % 2], mode='full')
            compensated.append(filtered)

        # Stack compensated impulse responses
        if len(compensated):
            impulse_responses = np.vstack(compensated)

    # Write standard channel order HRIR
    write_wav(os.path.join(out_dir, 'hrir.wav'), fs, impulse_responses)

    # Write HeSuVi channel order HRIR
    hesuvi_order = ['FL-left', 'FL-right', 'SL-left', 'SL-right', 'BL-left', 'BL-right', 'FC-left', 'FR-right',
                    'FR-left', 'SR-right', 'SR-left', 'BR-right', 'BR-left', 'FC-right']
    indices = [IR_ORDER.index(ch) for ch in hesuvi_order]
    write_wav(os.path.join(out_dir, 'hesuvi.wav'), fs, impulse_responses[indices, :])


def create_cli():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--measure', action='store_true',
                            help='Measure sine sweeps? Uses default audio output and input devices.')
    arg_parser.add_argument('--compensate_headphones', action='store_true',
                            help='Produce CSV file for AutoEQ from headphones sine sweep recordgin?')
    arg_parser.add_argument('--dir_path', type=str, help='Path to directory for recordings and outputs.')
    arg_parser.add_argument('--test_signal', type=str, help='File path to sine sweep test signal.')
    arg_parser.add_argument('--speakers', type=str,
                            help='Order of speakers in the recording as a comma separated list of speaker channel '
                                 'names. Supported names are "FL" (front left), "FR" (front right), '
                                 '"FC" (front center), "BL" (back left), "BR" (back right), '
                                 '"SL" (side left), "SR" (side right)". For example: "FL,FR".')
    arg_parser.add_argument('--silence_length', type=float,
                            help='Length of silence in the beginning, end and between recordings.')
    arg_parser.add_argument('--headphones', type=str,
                            help='File path to headphones sine sweep recording. Stereo WAV file is expected.')
    # TODO: filtfilt
    args = vars(arg_parser.parse_args())
    if 'speakers' in args and args['speakers'] is not None:
        args['speakers'] = args['speakers'].upper().split(',')
    return args


if __name__ == '__main__':
    main(**create_cli())
