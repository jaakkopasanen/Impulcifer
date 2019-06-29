# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
from autoeq.frequency_response import FrequencyResponse
from utils import magnitude_response


class ImpulseResponse:
    def __init__(self, data, fs, recording=None):
        self.fs = fs
        self.data = data
        self.recording = recording

    def peak_index(self):
        """Finds impulse reponse peak index

        Returns:
            Peak index to impulse response data
        """
        peaks, _ = signal.find_peaks(np.abs(self.data) / np.max(np.abs(self.data)), height=0.99)
        return peaks[0]

    def decay(self, window_size_ms=1.0):
        """Decay graph with RMS values for each window.

        Args:
            window_size_ms: RMS window size in milliseconds.

        Returns:
            RMS widows as Numpy array
        """
        # Sliding window RMS
        window_size = round(self.fs // 1000 * window_size_ms)

        # RMS windows
        n = len(self.data) // window_size
        windows = np.vstack(np.split(self.data[:n * window_size], n))
        rms = np.sqrt(np.mean(np.square(windows), axis=1))

        # Prevent division by zero in log10
        rms[rms == 0.0] = 1e-9

        return 20 * np.log10(rms)

    def tail_index(self):
        """Finds index after which there is nothing but noise.

        Returns:
            Index to impulse response data
        """
        # Generate decay data
        window_size_ms = 10
        rms = self.decay(window_size_ms=window_size_ms)

        # Smoothen data
        rms = signal.savgol_filter(rms, 21, 1)
        rms_peak_index = np.argmax(rms)

        # Find index where current RMS window value is greater than the previous indicating that noise floor has been
        # reached
        i = rms_peak_index + 1
        while i < len(rms):
            if rms[i] > rms[i - 1] - 0.1:  # 0.1dB threshold for improvement
                break
            i += 1

        # Tail index in impulse response, rms has larger window size
        tail_ind = i * window_size_ms * (self.fs / 1000)
        tail_ind = int(np.round(tail_ind))
        return tail_ind

    def plot_spectrogram(self, fig=None, ax=None, plot_file_path=None):
        """Plots spectrogram for a logarithmic sine sweep recording.

        Args:
            fig: Figure instance
            ax: Axis instance
            plot_file_path: Path to a file for saving the plot

        Returns:
            None
        """
        if self.recording is None or len(np.nonzero(self.recording)[0]) == 0:
            return
        if fig is None:
            fig, ax = plt.subplots()
        ax.specgram(self.recording, Fs=self.fs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        if plot_file_path:
            fig.savefig(plot_file_path)
        return fig, ax

    def plot_decay(self, window_size_ms=1, fig=None, ax=None, plot_file_path=None):
        """Plots decay graph.

        Args:
            window_size_ms: RMS window size in milliseconds.
            fig: Figure instance. New will be created if None is passed.
            ax: Axis instance. New will be created if None is passed to fig.
            plot_file_path: Save plot figure to a file.

        Returns:
            None
        """
        rms = self.decay(window_size_ms=window_size_ms)
        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(window_size_ms * np.arange(len(rms)), rms, linewidth=0.5)
        ax.set_ylim([-150, 0])
        ax.set_xlim([-100, len(rms) * window_size_ms])
        ax.set_xlabel('Time (ms)')
        ax.grid(True, which='major')
        ax.set_title('Decay')
        if plot_file_path:
            fig.savefig(plot_file_path)
        return fig, ax

    def plot_ir(self, fig=None, ax=None, max_time=None, plot_file_path=None):
        """Plots impulse response wave form.

        Args:
            fig: Figure instance
            ax: Axis instance
            max_time: Maximum time in seconds for cropping the tail.
            plot_file_path: Path to a file for saving the plot

        Returns:
            None
        """
        if max_time is None:
            max_time = len(self.data) / self.fs
        ir = self.data[:int(max_time * self.fs)]

        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(ir) / (self.fs / 1000), 1000 / self.fs), ir, linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.grid(True)
        ax.set_title('Impulse response {ms} ms'.format(ms=int(max_time * 1000)))

        if plot_file_path:
            fig.savefig(plot_file_path)

        return fig, ax

    def magnitude_response(self):
        return magnitude_response(self.data, self.fs)

    def plot_fr(self, fig=None, ax=None, plot_file_path=None, raw=True, smoothed=True):
        """Plots frequency response."""
        f, m = self.magnitude_response()
        fr = FrequencyResponse(name='Frequency response', frequency=f[1:], raw=m[1:])
        fr.interpolate()
        fr.smoothen_fractional_octave(
            window_size=1 / 3,
            iterations=1,
            treble_window_size=1 / 6,
            treble_iterations=1,
            treble_f_lower=100,
            treble_f_upper=1000,

        )
        if fig is None:
            fig, ax = plt.subplots()
        ax.set_xlabel('Frequency (Hz)')
        ax.semilogx()
        ax.set_xlim([20, 20e3])
        ax.set_ylabel('Amplitude (dBr)')
        ax.set_title(fr.name)
        ax.grid(True, which='major')
        ax.grid(True, which='minor')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        legend = []
        if raw:
            ax.plot(fr.frequency, fr.raw, linewidth=0.5)
            legend.append('Raw')
        if smoothed:
            ax.plot(fr.frequency, fr.smoothed, linewidth=1)
            legend.append('Smoothed')
        ax.legend(legend, fontsize=8)
        if plot_file_path:
            fig.savefig(plot_file_path)
        return fig, ax
