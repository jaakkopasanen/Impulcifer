# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import argparse


def fft(x, fs):
    nfft = len(x)
    df = fs / nfft
    f = np.arange(0, fs - df, fs / nfft)
    X = np.fft.fft(x)
    X_mag = 20 * np.log10(np.abs(X))
    return f[0:int(np.ceil(nfft/2))], X_mag[0:int(np.ceil(nfft/2))]


def normalize(x):
    dtype = x.dtype
    x = x.astype('float32')
    if dtype == 'int32':
        x /= 2.0 ** 31
    if dtype == 'int16':
        x /= 2.0 ** 15
    elif dtype == 'uint8':
        x /= 2.0 ** 8
        x *= 2.0
        x -= 1.0
    return x


def splice(x, fs, test_length, silence_length, n_speakers):
    # TODO: Use pydub
    # Splice data into multiple record + silence parts
    # There is silence in the beginning
    dtype = x.dtype
    sweeps = []
    for i in range(n_speakers):
        start = np.round(fs * (silence_length * (i + 1) + test_length * i)).astype('int')
        end = np.round(fs * (silence_length * (i + 2) + test_length * (i + 1))).astype('int')
        sweeps.append(x[start:end, 0])  # Left mic
        sweeps.append(x[start:end, 1])  # Right mic

    # Zero pad to longest sweep
    n = max([len(x) for x in sweeps])
    sweeps = np.vstack([np.concatenate((sweep, np.zeros(n - len(sweep)))) for sweep in sweeps])

    return sweeps.astype(dtype)


def xt(x, fs):
    return np.arange(0, len(x) / fs, 1 / fs)


def plot_sweep(x, fs, name=None):
    plt.plot(xt(x, fs), x)
    plt.xlim([0.0, len(x) / fs])
    plt.ylim([-1.0, 1.0])
    plt.ylabel('Amplitude')
    if name is not None:
        plt.legend([name])


def split(recording, test_signal, n_speakers=1, silence_length=2.0):
    """Splits sine sweep recording into individual speaker-ear pairs

    Signal looks something like this:
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
        test_signal: AudioSegement for the test signal
        n_speakers: Number of speakers per track
        silence_length: Length of silence in the beginning, end and between test signals

    Returns:

    """
    # TODO: Detect n_speakers from the recording

    columns = []
    ms = silence_length*1000 + len(test_signal)
    remainder = recording[silence_length*1000:]
    for _ in range(n_speakers):
        columns.append(remainder[:ms].split_to_mono())  # Add list of mono channel AudioSegments
        remainder = remainder[ms:]  # Cut to the beginning of next signal

    tracks = []
    i = 0
    while i < recording.channels:
        for column in columns:
            tracks.append(column[i])  # Left ear of current speaker
            tracks.append(column[i+1])  # Right ear of current speaker
        i += 2

    return tracks


# TODO: filtfilt
# TODO: Convolution
# TODO: Method for cropping impulse responses after convolution


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--recording', type=str, required=True, help='File path to sine sweep recording.')
    arg_parser.add_argument('--test', type=str, required=True, help='File path to sine sweep test signal.')
    arg_parser.add_argument('--n_speakers', type=int, required=True, help='Number of speakers in each track.')
    arg_parser.add_argument('--silence_length', type=float, required=True,
                            help='Length of silence in the beginning, end and between recordings.')
    cli_args = arg_parser.parse_args()

    # Read files
    rec = AudioSegment.from_wav(cli_args.recording)
    test = AudioSegment.from_wav(cli_args.test)

    print(len(rec))

    # Make sure sampling rates match
    if rec.frame_rate != test.frame_rate:
        raise ValueError('Sampling frequencies of test tone and recording must match!')

    # Split recording WAV file into individual mono tracks
    tracks = split(rec, test, n_speakers=cli_args.n_speakers, silence_length=cli_args.silence_length)

    for tr in tracks:
        print(len(tr))

    data = np.vstack([tr.get_array_of_samples() for tr in tracks])

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plot_sweep(
            normalize(data[i, :]),
            rec.frame_rate,
            name='Speaker {s:d} - {mic}'.format(s=int(i/2+1), mic='right' if i % 2 else 'left')
        )
    plt.xlabel('Time (s)')
    plt.show()

    if not os.path.isdir('out'):
        os.makedirs('out', exist_ok=True)

    AudioSegment.from_mono_audiosegments(*tracks).export('out/sweeps.wav', format='wav')
    AudioSegment.from_mono_audiosegments(*tracks).export('out/tests.wav', format='wav')


if __name__ == '__main__':
    main()
