# -*- coding: utf-8 -*-

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.mlab import specgram
from matplotlib import cm
from scipy import signal
import nnresample
from autoeq.frequency_response import FrequencyResponse
from utils import magnitude_response


class ImpulseResponse:
    def __init__(self, data, fs, recording=None):
        self.fs = fs
        self.data = data
        self.recording = recording

    def __len__(self):
        """Impulse response length in samples."""
        return len(self.data)

    def duration(self):
        """Impulse response duration in seconds."""
        return len(self) / self.fs

    def active_duration(self):
        """Impulse response duration from peak to noise in seconds."""
        return (self.tail_index() - self.peak_index()) / self.fs

    def peak_index(self, start=0, end=None, peak_height=0.67):
        """Finds the first high (negative or positive) peak in the impulse response wave form.

        Returns:
            Peak index to impulse response data
        """
        if end is None:
            end = len(self.data)
        # Peak height threshold, relative to the data maximum value
        # Copy to avoid manipulating the original data here
        data = self.data.copy()
        # Limit search to given range
        data = data[start:end]
        # Normalize to 1.0
        data /= np.max(np.abs(data))
        # Find positive peaks
        peaks_pos, properties = signal.find_peaks(data, height=peak_height)
        # Find negative peaks that are at least
        peaks_neg, _ = signal.find_peaks(data * -1.0, height=peak_height)
        # Combine positive and negative peaks
        peaks = np.concatenate([peaks_pos, peaks_neg])
        # Add start delta to peak indices
        peaks += start
        # Return the first one
        return np.min(peaks)

    def decay(self):
        """Decay graph with RMS values for each window.

        Returns:
            RMS widows as Numpy array
        """
        # Envelope
        analytical_signal = self.data
        amplitude_envelope = np.abs(analytical_signal)
        # dB scale and normalized to 0 dB
        amplitude_envelope /= np.max(amplitude_envelope)
        decay = amplitude_envelope

        inds, _ = signal.find_peaks(decay)
        heights = decay[inds]
        n = int(len(inds) / self.duration() * 0.05)  # Arbitrary 0.05 to scale the peak selection window size
        top = inds[0]  # Index of top value in peak indexes
        tops = [top]
        while top + 1 < len(heights):
            top = np.argmax(heights[top + 1:top + 1 + n]) + top + 1
            tops.append(top)
        skyline = inds[tops]  # Indexes to decay

        decay = decay[skyline]
        return skyline, decay

    def tail_index(self):
        """Finds index after which there is nothing but noise.

        Returns:
            Index to impulse response data
        """
        # Generate decay data
        inds, decay = self.decay()

        # Find starting point
        peak_index = np.argmax(decay)

        # Find index where current value is greater than the previous indicating that noise floor has been
        # reached
        i = peak_index + 1
        while i < len(decay):
            if decay[i] > decay[i - 1]:  # 0.1dB threshold for improvement
                break
            i += 1

        # Tail index in impulse response, rms has larger window size
        return inds[i]

    def equalize(self, fir):
        """Equalizes this impulse response with give FIR filter.

        Args:
            fir: FIR filter as an single dimensional array

        Returns:
            None
        """
        self.data = signal.convolve(self.data, fir, mode='full')

    def resample(self, fs):
        """Resamples this impulse response to the given sampling rate."""
        self.data = nnresample.resample(self.data, fs, self.fs)
        self.fs = fs

    def pnr(self):
        """Calculates peak to noise ratio"""
        data = self.data / np.max(np.abs(self.data))  # Normalize to 0 dB
        tail_index = self.tail_index()  # Index where IR decays to noise floor
        # Select one second after the IR decays to noise floor as noise sample
        tail = data[tail_index:tail_index + self.fs]
        # 0 dB over mean squares
        return 10 * np.log10(1 / (1 / len(tail) * np.sum(tail ** 2)))

    def plot_spectrogram(self, fig=None, ax=None, plot_file_path=None, f_res=10, n_segments=200):
        """Plots spectrogram for a logarithmic sine sweep recording.

        Args:
            fig: Figure instance
            ax: Axis instance
            plot_file_path: Path to a file for saving the plot
            f_res: Frequency resolution (step) in Hertz
            n_segments: Number of segments in time axis

        Returns:
            None
        """
        if self.recording is None or len(np.nonzero(self.recording)[0]) == 0:
            return
        if fig is None:
            fig, ax = plt.subplots()

        # Window length in samples
        nfft = int(self.fs / f_res)
        # Overlapping in samples
        noverlap = int(nfft - (len(self.recording) - nfft) / n_segments)
        # Get spectrogram data
        spectrum, freqs, t = specgram(self.recording, Fs=self.fs, NFFT=nfft, noverlap=noverlap, mode='psd')

        # Remove zero frequency
        f = freqs[1:]
        z = spectrum[1:, :]
        # Logarithmic power
        z = 10 * np.log10(z)

        # Create spectrogram image
        t, f = np.meshgrid(t, f)
        cs = plt.pcolormesh(t, f, z, cmap=cm.gnuplot2)
        fig.colorbar(cs)
        ax.semilogy()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')

        # Save image
        if plot_file_path:
            fig.savefig(plot_file_path)

        return fig, ax

    def plot_decay(self, fig=None, ax=None, plot_file_path=None):
        """Plots decay graph.

        Args:
            fig: Figure instance. New will be created if None is passed.
            ax: Axis instance. New will be created if None is passed to fig.
            plot_file_path: Save plot figure to a file.

        Returns:
            None
        """
        if fig is None:
            fig, ax = plt.subplots()
        t, decay = self.decay()
        ax.plot(t / self.fs * 1000, 20 * np.log10(decay), linewidth=1)

        ax.set_ylim([None, 10])
        ax.set_xlim([0, len(self) / self.fs * 1000])
        ax.set_xlabel('Time (ms)')
        ax.grid(True, which='major')
        ax.set_title('Decay')
        if plot_file_path:
            fig.savefig(plot_file_path)
        return fig, ax

    def plot_ir(self, fig=None, ax=None, start=0.0, end=None, plot_file_path=None):
        """Plots impulse response wave form.

        Args:
            fig: Figure instance
            ax: Axis instance
            start: Start of the plot in seconds
            end: End of the plot in seconds
            plot_file_path: Path to a file for saving the plot

        Returns:
            None
        """
        if end is None:
            end = len(self.data) / self.fs
        ir = self.data[int(start * self.fs):int(end * self.fs)]

        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(np.arange(start * 1000, start * 1000 + 1000 / self.fs * len(ir), 1000 / self.fs), ir, linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.set_title('Impulse response'.format(ms=int(end * 1000)))

        if plot_file_path:
            fig.savefig(plot_file_path)

        return fig, ax

    def magnitude_response(self):
        """Calculates magnitude response for the data."""
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
