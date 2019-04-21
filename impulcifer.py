# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import ticker
from pydub import AudioSegment
import argparse
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import fftconvolve, kaiser
import pyfftw
from time import time

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
SPEAKER_DELAYS = {
    'FL': 0.1487,
    'FR': 0.1487,
    'FC': 0.2557,
    'BL': 0.1487,  # TODO: Confirm this
    'BR': 0.1487,  # TODO: Confirm this
    'SL': 0.0417,
    'SR': 0.0417,
}


def magnitude_response(x, fs):
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


def impulse_response_decay(impulse_response, fs, window_size_ms=1, show_plot=False, plot_file_path=None):
    # Sliding window RMS
    window_size = fs // 1000 * window_size_ms

    # RMS windows
    n = len(impulse_response) // window_size
    windows = np.vstack(np.split(impulse_response[:n * window_size], n))
    rms = np.sqrt(np.mean(np.square(windows), axis=1))

    # Smoothen data
    smoothed = 20*np.log10(rms)
    for _ in range(200):
        smoothed = signal.savgol_filter(smoothed, 11, 1)

    if show_plot or plot_file_path:
        fig, ax = plt.subplots()
        plt.plot(window_size_ms * np.arange(len(rms)), 20*np.log10(rms))
        plt.plot(window_size_ms * np.arange(len(rms)), smoothed)
        plt.ylim([-150, 0])
        plt.xlim([-100, len(rms) * window_size_ms])
        plt.xlabel('Time (ms)')
        plt.grid(True, which='major')
        if show_plot:
            plt.show()
        if plot_file_path:
            plt.savefig(plot_file_path)

    return rms


def tail_index(rms, rms_window_size):
    """Finds index in an impulse response after which there is nothing but noise left

    Args:
        rms: RMS values for windows as numpy array
        rms_window_size: Number of sample in each RMS window

    Returns:
        Tail index
    """
    # Peak
    peak_index = np.argmax(rms)

    for i in range(peak_index+1, len(rms)):
        if rms[i] > rms[i-1]:
            break
    noise_floor = rms[i]
    noise_floor *= 2  # +6dB headroom
    tail_ind = np.argmax(rms < noise_floor)

    # Tail index in impulse response, rms has larger window size
    tail_ind *= rms_window_size

    return int(tail_ind)


def crop_ir_head(left, right, speaker, fs):
    """Crops out silent head of left and right ear impulse responses and sets delay to correct value according to
    speaker channel

    Args:
        left: AudioSegment for left ear impulse response
        right: AudioSegment for right ear impulse response
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
    delay = int(np.round(SPEAKER_DELAYS[speaker] / 1000 * fs))  # Channel delay in samples
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
    # TODO: check if this is necessary
    left[0] *= 0.0
    left[1] *= 0.5
    right[0] *= 0.0
    right[1] *= 0.5

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


def spectrogram(sweep, fs, show_plot=False, plot_file_path=None):
    if len(np.nonzero(sweep)[0]) == 0:
        return

    fig, ax = plt.subplots()
    plt.specgram(sweep, Fs=fs)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    if show_plot:
        plt.show()
    if plot_file_path:
        plt.savefig(plot_file_path)


def main(measure=False,
         equalize=False,
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

    # Logarithmic sine sweep measurement
    if measure:  # TODO
        raise NotImplementedError('Measurement is not yet implemented.')

    # Split recording WAV file into individual mono tracks
    recording = split_recording(recording, test_signal, speakers, fs, silence_length)

    # Reorder tracks to match standard
    recording = reorder_tracks(recording, speakers)

    # Normalize to -0.1 dB
    recording = normalize(recording, target_db=-0.1)

    for i in range(recording.shape[0]):
        spectrogram(recording[i, :], fs, plot_file_path=os.path.join(out_dir, 'spectrogram_{}.png'.format(IR_ORDER[i])))

    # Write multi-channel WAV file with sine sweeps for debugging
    write_wav(os.path.join(out_dir, 'preprocessed.wav'), fs, recording)

    # Pad test signal to pre-processed recording length with zeros
    test_signal = zero_pad(test_signal, recording.shape[1])

    # Estimate impulse responses by deconvolution
    impulse_responses = []
    for i in range(recording.shape[0]):
        track = recording[i, :]
        if len(np.nonzero(track)[0]) > 0:
            # Run deconvolution
            impulse_response = deconv(
                track,
                test_signal,
                method='inverse_filter',
                fs=fs
            )

            # Add to responses
            impulse_responses.append(impulse_response)
        else:
            # Silent track
            impulse_responses.append(np.zeros(recording.shape[1]))
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
                plot_file_path=os.path.join(out_dir, 'decay_{}_.png'.format(IR_ORDER[i]))
            )
            tail_indices.append(tail_index(rms, rms_window_size=fs / 1000))
    # Crop all tracks by last tail index
    tail_ind = max(tail_indices)
    for i in range(len(cropped)):
        cropped[i] = cropped[i][:tail_ind]
    impulse_responses = np.vstack(cropped)

    # Write standard channel order HRIR
    write_wav(os.path.join(out_dir, 'hrir.wav'), fs, impulse_responses)

    # Write HeSuVi channel order HRIR
    hesuvi_order = ['FL-left', 'FL-right', 'SL-left', 'SL-right', 'BL-left', 'BL-right', 'FC-left', 'FR-right',
                    'FR-left', 'SR-right', 'SR-left', 'BR-right', 'BR-left', 'FC-right']
    indices = [IR_ORDER.index(ch) for ch in hesuvi_order]
    write_wav(os.path.join(out_dir, 'hesuvi.wav'), fs, impulse_responses[indices, :])

    if equalize:
        # Read WAV file
        if type(headphones) == str:
            headphones = AudioSegment.from_wav(headphones)
        for i, ch in enumerate(headphones.split_to_mono()):
            f, m = magnitude_response(ch.get_array_of_samples(), headphones.frame_rate)
            track = ['frequency,raw']
            for j in range(len(f)):
                if f[j] > 0.0:
                    track.append('{f:.2f},{m:.2f}'.format(f=f[j], m=m[j]))
            with open(os.path.join(out_dir, 'headphones-{}.csv'.format('left' if i == 0 else 'right')), 'w') as file:
                file.write('\n'.join(track))


def create_cli():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--measure', action='store_true',
                            help='Measure sine sweeps? Uses default audio output and input devices.')
    arg_parser.add_argument('--equalize', action='store_true',
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
