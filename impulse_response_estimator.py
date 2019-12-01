# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import pickle
from scipy.fftpack import fft
from scipy.signal import convolve
from scipy.signal.windows import hann
import numpy as np
import matplotlib.pyplot as plt
from utils import read_wav, write_wav, magnitude_response


class ImpulseResponseEstimator(object):
    """
    Gives probe impulse, gets recording and calculate impulse response.
    Method from paper: "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
    Angelo Farina
    """

    def __init__(self, min_duration=5.0, fs=44100):
        if fs != int(fs):
            raise ValueError('Sampling rate "fs" must be an integer.')
        self.fs = int(fs)
        # End frequency is always Nyquist frequency
        self.high = self.fs / 2
        # Start frequency is less than 5 Hz but scaled in a way that there is integer number of octaves between
        # start and end frequencies
        self.low = 5
        self.n_octaves = np.ceil(np.log2(self.high / self.low))  # P
        self.low = self.high / 2**self.n_octaves

        # Total length in samples
        self.w1 = self.low / self.fs * 2 * np.pi
        self.w2 = self.high / self.fs * 2 * np.pi

        # Generate test signal
        self.test_signal = self.generate_test_signal(min_duration)
        self.duration = len(self.test_signal) / self.fs

        # Generate inverse filter
        self.inverse_filter = self.generate_inverse_filter()

    def __len__(self):
        return len(self.test_signal)

    def plot(self):
        f, m = magnitude_response(self.test_signal, self.fs)
        plt.plot(f, m)
        f, m = magnitude_response(self.inverse_filter, self.fs)
        plt.plot(f, m)
        f, m = magnitude_response(self.estimate(self.test_signal), self.fs)
        plt.plot(f, m)
        plt.semilogx()
        plt.legend(['Test signal spectrum', 'Inverse filter spectrum', 'Impulse response spectrum'])
        plt.grid(True, which='major')
        plt.grid(True, which='minor')
        plt.show()

        plt.plot(np.arange(len(self)) / self.fs, self.estimate(self.test_signal))
        plt.title('Impulse response')
        plt.grid(True)
        plt.show()

    def generate_inverse_filter(self):
        """Generates inverse filter for test signal.

        Returns:

        """
        P = self.n_octaves
        N = len(self.test_signal)
        inverse_filter = np.flip(self.test_signal) * (2**(P / N))**(np.arange(N)*-1) * P * np.log(2) / (1 - 2**-P)

        # Now we have to normalize energy of result of dot product.
        # This is "naive" method but it just works.
        frp = fft(convolve(inverse_filter, self.test_signal, method='auto'))
        inverse_filter /= np.abs(frp[round(frp.shape[0]/4)])

        return inverse_filter

    def generate_test_signal(self, min_duration, fade_in=1/2, fade_out=None):
        """Generates test signal.

        Simultaneous Measurement of Impulse Response and Distortion with a Swept-Sine Technique.
        Angelo Farina, 2000
        http://pcfarina.eng.unipr.it/Public/Papers/134-AES00.PDF

        Advancements in impulse response measurements by sine sweeps
        Angelo Farina, 2007
        http://pcfarina.eng.unipr.it/Public/Papers/226-AES122.pdf

        Optimizing the exponential sine sweep (ESS) signal for in situ measurements on noise barriers
        Massimo Garai and Paolo Guidorzi, 2015
        https://www.researchgate.net/publication/280131468_Optimizing_the_exponential_sine_sweep_ESS_signal_for_in_situ_measurements_on_noise_barriers

        Args:
            min_duration: Minimum test signal duration in seconds.
            fade_in: Size of fade-in Hanning window in octaves. None value disables fade-in.
            fade_out: Size of fade-out Hanning window in octaves. None value disables fade-out.

        Returns:
            Test signal
        """
        # P is the number of octaves in the test signal
        P = self.n_octaves
        # M is a length multiplier
        # See equation 2 in Garai and Guidorzi 2015
        # Here it's selected such that the test signal duration is equal or greater than minimum duration
        M = np.ceil(min_duration * self.fs * (np.pi / 2**P) / (np.pi * 2 * np.log(2**P)))
        # L is the real number length of the test signal in samples
        L = M * np.pi * 2 * np.log(2**P) / (np.pi / 2**P)
        # N is the actual length of the test signal in samples
        N = np.round(L)
        freqs = np.pi / 2**P * L / np.log(2**P) * np.exp(np.arange(N) / N * np.log(2**P))
        test_signal = np.sin(freqs)

        seconds_per_octave = N / self.fs / P

        # Fade-in window
        if fade_in is None:
            fade_in_window = []
        else:
            fade_in = 2 * int(self.fs * seconds_per_octave * fade_in)
            if fade_in % 2:
                fade_in += 1
            fade_in_window = hann(fade_in)[:fade_in // 2]

        # Fade-out window
        if fade_out is None:
            fade_out_window = []
        else:
            fade_out = 2 * int(self.fs * seconds_per_octave * fade_out)
            if fade_out % 2:
                fade_out += 1
            fade_out_window = hann(fade_out)[fade_out // 2:]

        # Create window from fade-in window and fade-out window with ones in the middle
        win = np.concatenate([
            fade_in_window,
            np.ones(len(test_signal) - len(fade_in_window) - len(fade_out_window)),
            fade_out_window
        ])
        test_signal *= win

        return test_signal

    def estimate(self, recording):
        """Estimates impulse response"""
        return convolve(recording, self.inverse_filter, mode='same', method='auto')

    def sweep_sequence(self, speakers, tracks):
        """Creates sine sweep sequence data with multiple tracks

        Output depends on the speakers and tracks in a way that speakers define which physical speakers will should be
        used during the playback and tracks define how many tracks the the output should have.

        When speakers is [FL, FR] and tracks is "stereo" the output will contain two tracks of which the first is the FL
        and the second is FR. In this case the FL will play first and FR track will be silent and the FR will play while
        the FL is silent.

        When speakers is [FL, FR] and tracks is "7.1" the output will contain 8 tracks of which first two are FL and FR.
        All other tracks in the output data are silent throughout the whole sequence. When speakers is [FL, FC, FR, SR,
        BR, BL, SL] there will be 7 tracks with sine sweep and one silent track (LFE) and physical speakers will play
        the sweeps in the order given in speakers.

        Mono sequences can be made by single speaker name in speakers. In most cases tracks should not be "mono" when
        playing back mono sequences because the single track might get upmixed to stereo. In most systems it's required
        to have tracks as "stereo" or higher when playing mono sequences. Playing mono sweep on center channel could
        be achieved by setting speakers to [FC] and tracks to "7.1" or "5.1". Playing mono sweep on left channel could
        be achieved by setting speakers to [FL] and tracks to "stereo" (or higher depending on the speaker system).

        Args:
            speakers: List of speaker names to use in the sequence
            tracks: Tracks configuration. "7.1", "5.1", "stereo" or "mono".

        Returns:
            Sweep sequence data as Numpy array. Each row represents a single track.
        """
        unique_speakers = []
        for speaker in speakers:
            if speaker in unique_speakers:
                raise ValueError('All speaker names in speakers must be unique.')

        # Remap channels
        if tracks == '7.1':
            standard_order = 'FL FR FC LFE BL BR SL SR'.split()
            n_tracks = 8
        elif tracks == '5.1':
            standard_order = 'FL FR FC LFE BL BR'.split()
            n_tracks = 6
        elif tracks == 'stereo':
            standard_order = 'FL FR'.split()
            n_tracks = 2
        elif tracks == 'mono':
            standard_order = ['FL']
            speakers = ['FL']
            n_tracks = 1
        else:
            raise ValueError('Unsupported track configuration "{}".'.format(tracks))

        for speaker in speakers:
            if speaker not in standard_order:
                raise ValueError('Speaker name "{speaker}" not supported with track configuration "{tracks}"'.format(
                    speaker=speaker,
                    tracks=tracks
                ))
        speaker_indices = [standard_order.index(ch) for ch in speakers]

        # Create test signal sequence
        data = np.zeros((n_tracks, int((self.fs * 2.0 + len(self)) * len(speakers) + self.fs * 2.0)))
        for i, speaker in enumerate(speakers):
            start_zeros = int((self.fs * 2.0 + len(self)) * i + self.fs * 2.0)
            end_zeros = int((self.fs * 2.0 + len(self)) * (len(speakers) - i - 1) + self.fs * 2.0)
            sweep_padded = np.concatenate([
                np.zeros((start_zeros,)),
                self.test_signal,
                np.zeros((end_zeros,))
            ])
            data[speaker_indices[i], :] = sweep_padded
        data = np.vstack(data)

        return data

    @classmethod
    def from_wav(cls, file_path):
        """Creates ImpulseResponseEstimator instance from test signal WAV."""
        fs, data = read_wav(file_path)
        ire = cls(min_duration=(len(data) - 1) / fs, fs=fs)
        if np.max(ire.test_signal - data) > 1e-9:
            raise ValueError('Data read from WAV file does not match generated test signal. WAV file must be generated '
                             'with the current version of ImpulseResponseEstimator.')
        return ire

    @staticmethod
    def from_pickle(file_path):
        """Creates ImpulseResponseEstimator instance from pickled file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def to_pickle(self, file_path):
        """Saves self to pickled file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def file_name(self, bit_depth):
        """Formats a file name for test signal without prefixe or file format

        Args:
            bit_depth: Bit depth of the target WAV file

        Returns:
            File name name
        """
        return f'{self.duration:.2f}s-{self.fs:d}Hz-{bit_depth:d}bit-{self.low:.2f}Hz-{self.high:.0f}Hz'


def create_cli():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dir_path', type=str, required=True,
                            help='Path to directory where generated test signal is saved. Default file name is used.')
    arg_parser.add_argument('--fs', type=int, required=True,
                            help='Sampling rate in Hertz.')
    arg_parser.add_argument('--duration', type=float, required=False, default=5.0,
                            help='Test signal duration in seconds. Defaults to 5.0 seconds.')
    arg_parser.add_argument('--bit_depth', type=int, required=False, default=32,
                            help='Test signal WAV file bit depth. Defaults to 32 bits.')
    arg_parser.add_argument('--speakers', type=str, required=False, default='FL',
                            help='Speaker channel order in test signal sequence as comma separated values. Mono test '
                                 'signal sequence is created by default but if speakers are specified a second sequence'
                                 ' file will be generated which contains multiple sweeps separated by silences in the '
                                 'order of given speaker channels. Stereo sequence can be generated by supplying value '
                                 '"FL,FR". Supported channel names are "FL", "FR", "FC", "SL", "SR", "BL" and "BR".')
    arg_parser.add_argument('--tracks', type=str, required=False, default='mono',
                            help='WAV file track configuration. Supported values are "mono", "stereo", "5.1" and '
                                 '"7.1". This should be set according to sound card. Supported speaker names for '
                                 '"stereo" are "FL" and "FR". Supported speaker names for "7.1" are "FL", "FR", "FC", '
                                 '"BL" and "BR". Supported speaker names for "5.1" are "FL", "FR", "FC", '
                                 '"SL", "SR", "BL" and "BR". "mono" will force speakers to "FL".')
    cli_args = arg_parser.parse_args()
    if not os.path.isdir(cli_args.dir_path):
        # File path is required
        raise TypeError('--dir_path must be a directory.')
    return cli_args


def main():
    # Handle command line arguments
    cli_args = create_cli()
    dir_path = cli_args.dir_path
    fs = cli_args.fs
    duration = cli_args.duration
    bit_depth = cli_args.bit_depth
    speakers = cli_args.speakers.split(',')
    tracks = cli_args.tracks

    # Create sweep sequence WAV data
    ire = ImpulseResponseEstimator(min_duration=duration, fs=fs)
    wav_data = ire.sweep_sequence(speakers, tracks)

    # Write test signal to WAV file
    file_name = f'sweep-{ire.file_name(bit_depth)}.wav'
    write_wav(os.path.join(dir_path, file_name), ire.fs, ire.test_signal, bit_depth=bit_depth)

    # Write test signal to pickle file
    file_name = f'sweep-{ire.file_name(bit_depth)}.pkl'
    ire.to_pickle(os.path.join(dir_path, file_name))

    # Write test signal sequence to WAV file
    file_name = f'sweep-seg-{",".join(speakers)}-{tracks}-{ire.file_name(bit_depth)}.wav'
    write_wav(os.path.join(dir_path, file_name), fs, wav_data, bit_depth=bit_depth)


if __name__ == '__main__':
    main()
