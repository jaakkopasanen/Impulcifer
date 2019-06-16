# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
from scipy.fftpack import fft
from scipy.signal import fftconvolve, kaiser, hanning
import numpy as np
import impulcifer


class ImpulseResponseEstimator(object):
    """
    Gives probe impulse, gets recording and calculate impulse response.
    Method from paper: "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
    Angelo Farina
    """

    def __init__(self, duration=5.0, fs=44100):
        if fs != int(fs):
            raise ValueError('Sampling rate "fs" must be an integer.')
        self.fs = int(fs)
        self.low = 5.0  # Start frequency
        self.high = self.fs / 2  # End frequency is always Nyquist frequency

        # Total length in samples
        self.T = fs*duration
        self.w1 = self.low / self.fs * 2*np.pi
        self.w2 = self.high / self.fs * 2*np.pi

        # Generate test signal
        self.test_signal = self.generate_test_signal()

        # This is what the value of K will be at the end (in dB):
        kend = 10**((-6*np.log2(self.w2/self.w1))/20)
        # dB to rational number.
        k = np.log(kend)/self.T

        # Making reverse probe impulse so that convolution will just calculate dot product.
        # Weighting it with exponent to achieve 6 dB per octave amplitude decrease.
        c = np.array(list(map(lambda t: np.exp(float(t)*k), range(int(self.T)))))
        self.inverse_filter = np.flip(self.test_signal) * c

        # Now we have to normalize energy of result of dot product.
        # This is "naive" method but it just works.
        frp = fft(fftconvolve(self.inverse_filter, self.test_signal))
        self.inverse_filter /= np.abs(frp[round(frp.shape[0]/4)])

    def generate_test_signal(self):
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

        Returns:
            Test signal
        """
        # TODO: Phase synced sine sweep as described in Garai and Guidorzi
        w1 = self.w1
        w2 = self.w2
        T = self.T

        def log_freq(t):
            K = T * w1 / np.log(w2/w1)
            L = T / np.log(w2/w1)
            return K * (np.exp(t/L)-1.0)

        freqs = log_freq(range(int(T)))
        impulse = np.sin(freqs)

        def seconds_per_octave(octaves):
            return octaves / (np.log2(self.high / self.low) / (self.T * self.fs))

        # Fade-in window
        fade_in_window_len = 2 * int(self.fs * seconds_per_octave(1/2))
        if fade_in_window_len % 2:
            fade_in_window_len += 1
        fade_in_window = hanning(fade_in_window_len)[:fade_in_window_len // 2]

        # Fade-out window
        fade_out_window_len = 2 * int(self.fs * seconds_per_octave(1/24))
        if fade_out_window_len % 2:
            fade_out_window_len += 1
        fade_out_window = hanning(fade_out_window_len)[:fade_out_window_len // 2]

        # Create window from fade-in window and fade-out window with ones in the middle
        win = np.concatenate([
            fade_in_window,
            np.ones(len(impulse) - len(fade_in_window) - len(fade_out_window)),
            fade_out_window
        ])
        impulse *= win
        return impulse

    def estimate(self, recording):
        """Estimates impulse response"""
        ir = fftconvolve(recording, self.inverse_filter, mode='full')
        ir = ir[self.test_signal.shape[0]:self.test_signal.shape[0]*2+1]
        return ir

    @classmethod
    def from_wav(cls, file_path):
        """Creates ImpulseResponseEstimator instance from test signal WAV."""
        fs, data = impulcifer.read_wav(file_path)
        ire = cls(duration=len(data) / fs, fs=fs)
        if np.max(ire.test_signal, data) > 1e-9:
            raise ValueError('Data read from WAV file does not match generated test signal. WAV file must be generated '
                             'with the current version of ImpulseResponseEstimator.')
        return ire


def main():
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
                                 'signal sequence is always created but if speakers are specified a second sequence '
                                 'file will be generated which contains multiple sweeps separated by silences in the '
                                 'order of given speaker channels. Stereo sequence can be generated by supplying value '
                                 '"FL,FR". Supported channel names are "FL", "FR", "FC", "SL", "SR", "BL" and "BR".')
    cli_args = arg_parser.parse_args()
    if not os.path.isdir(cli_args.dir_path):
        # File path is required
        raise TypeError('--dir_path must be a directory.')

    speakers = cli_args.speakers.split(',')

    # Create instance
    ire = ImpulseResponseEstimator(duration=cli_args.duration, fs=cli_args.fs)

    # Create file path with directory and default file name
    file_path = os.path.join(cli_args.dir_path, 'sweep-{t}s-{fs:d}Hz-{bits:d}bit-{low}Hz-{high:.1f}Hz.wav'.format(
        fs=cli_args.fs, t=cli_args.duration, bits=cli_args.bit_depth, low=ire.low, high=ire.high
    ))
    speaker_seq = ','.join(speakers) if len(speakers) > 1 else 'M'
    seq_file_path = file_path.replace('sweep-', 'sweep-seg-{}-'.format(speaker_seq))

    # Write test signal to WAV file
    impulcifer.write_wav(file_path, ire.fs, ire.test_signal, bit_depth=cli_args.bit_depth)

    # Create test signal sequence
    data = []
    for i, speaker in enumerate(speakers):
        start_zeros = int((ire.fs * 2.0 + ire.T) * i + ire.fs * 2.0)
        end_zeros = int((ire.fs * 2.0 + ire.T) * (len(speakers) - i - 1) + ire.fs * 2.0)
        data.append(np.concatenate([
            np.zeros((start_zeros,)),
            ire.test_signal,
            np.zeros((end_zeros,))
        ]))
    data = np.vstack(data)
    # Write test signal sequence
    # TODO: Channel mapping
    impulcifer.write_wav(seq_file_path, ire.fs, data, bit_depth=cli_args.bit_depth)


if __name__ == '__main__':
    main()
