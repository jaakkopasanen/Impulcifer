# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.mlab import specgram
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal.windows import hann
from scipy.ndimage import uniform_filter
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
            - Indexes to decay data
            - Values of decay data
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

        # Find index where current value is greater than the previous indicating that noise floor has been reached
        i = peak_index + 1
        while i < len(decay):
            if decay[i] > decay[i - 1]:
                break
            i += 1

        # Tail index in impulse response, rms has larger window size
        if i >= len(inds):
            return len(self.data) - 1
        return inds[i]

    def crop_head(self, head_ms=1):
        """Crops away head."""
        self.data = self.data[self.peak_index() - int(self.fs * head_ms / 1000):]

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

    def reverberation_time(self, target_level=-60):
        """Calculates reverberation time until a given attenuation level has been reached.

        Args:
            target_level: Target attenuation level in dB

        Returns:
            Reverberation time in seconds
        """
        inds, decay = self.decay()
        decay = 20 * np.log10(decay)
        peak_ind = np.argmax(decay)
        tail_ind = self.tail_index()
        for i in range(peak_ind, len(decay)):
            if decay[i] < target_level:
                return (inds[i] - inds[peak_ind]) / self.fs
            if inds[i] / self.fs > tail_ind:
                return (tail_ind - inds[peak_ind]) / self.fs

    def magnitude_response(self):
        """Calculates magnitude response for the data."""
        return magnitude_response(self.data, self.fs)

    def frequency_response(self):
        """Creates FrequencyResponse instance."""
        f, m = self.magnitude_response()
        n = self.fs / 2 / 4  # 4 Hz resolution
        step = int(len(f) / n)
        fr = FrequencyResponse(name='Frequency response', frequency=f[1::step], raw=m[1::step])
        fr.interpolate()
        return fr

    def plot(self,
             fig=None,
             ax=None,
             plot_file_path=None,
             plot_recording=True,
             plot_spectrogram=True,
             plot_ir=True,
             plot_fr=True,
             plot_decay=True,
             plot_waterfall=True):
        """Plots all plots into the same figure

        Args:
            fig: Figure instance
            ax: Axes instance, must have 2 rows and 3 columns
            plot_file_path: Path to a file for saving the plot
            plot_recording: Plot recording waveform?
            plot_spectrogram: Plot recording spectrogram?
            plot_ir: Plot impulse response?
            plot_fr: Plot frequency response?
            plot_decay: Plot decay curve?
            plot_waterfall: Plot waterfall graph?

        Returns:
            Figure
        """
        if fig is None:
            # Create figure and axises for the plots
            fig = plt.figure()
            ax = []
            for i in range(5):
                ax.append(fig.add_subplot(2, 3, i + 1))
            ax.append(fig.add_subplot(2, 3, 6, projection='3d'))
            ax = np.vstack([ax[:3], ax[3:]])
        if plot_recording:
            self.plot_recording(fig=fig, ax=ax[0, 0])
        if plot_spectrogram:
            self.plot_spectrogram(fig=fig, ax=ax[1, 0])
        if plot_ir:
            self.plot_ir(fig=fig, ax=ax[0, 1])
        if plot_fr:
            self.plot_fr(fig=fig, ax=ax[1, 1])
        if plot_decay:
            self.plot_decay(fig=fig, ax=ax[0, 2])
        if plot_waterfall:
            self.plot_waterfall(fig=fig, ax=ax[1, 2])
        if plot_file_path:
            fig.savefig(plot_file_path)
        return fig

    def plot_recording(self, fig=None, ax=None, plot_file_path=None):
        """Plots recording wave form

        Args:
            fig: Figure instance
            ax: Axes instance
            plot_file_path: Path to a file for saving the plot

        Returns:
            - Figure
            - Axes
        """
        if self.recording is None or len(np.nonzero(self.recording)[0]) == 0:
            return
        if fig is None:
            fig, ax = plt.subplots()

        ax.plot(np.linspace(0, len(self.recording) / self.fs, len(self.recording)), self.recording, linewidth=0.5)

        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Sine Sweep')

        # Save image
        if plot_file_path:
            fig.savefig(plot_file_path)

        return fig, ax

    def plot_spectrogram(self, fig=None, ax=None, plot_file_path=None, f_res=10, n_segments=200):
        """Plots spectrogram for a logarithmic sine sweep recording.

        Args:
            fig: Figure instance
            ax: Axis instance
            plot_file_path: Path to a file for saving the plot
            f_res: Frequency resolution (step) in Hertz
            n_segments: Number of segments in time axis

        Returns:
            - Figure
            - Axis
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
        cs = ax.pcolormesh(t, f, z, cmap='gnuplot2')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cs, cax=cax)

        ax.semilogy()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')

        # Save image
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

    def plot_fr(self,
                fr=None,
                fig=None,
                ax=None,
                plot_file_path=None,
                plot_raw=True,
                plot_smoothed=True,
                plot_error=True,
                plot_error_smoothed=True,
                plot_target=True,
                plot_equalization=True,
                plot_equalized=True):
        """Plots frequency response

        Args:
            fr: FrequencyResponse instance. Useful for passing instance with taget, error, equalization etc...
            fig: Figure instance
            ax: Axes instance
            plot_file_path: Path to a file for saving the plot
            plot_raw: Include raw curve?
            plot_smoothed: Include smoothed curve?
            plot_error: Include unsmoothed error curve?
            plot_error_smoothed: Include smoothed error curve?
            plot_target: Include target curve?
            plot_equalization: Include equalization curve?
            plot_equalized: Include equalized curve?

        Returns:
            - Figure
            - Axes
        """
        if fr is None:
            fr = self.frequency_response()
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
        ax.set_ylabel('Amplitude (dB)')
        ax.set_title(fr.name)
        ax.grid(True, which='major')
        ax.grid(True, which='minor')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        legend = []

        if plot_target and len(fr.target):
            ax.plot(fr.frequency, fr.target, linewidth=5, color='lightblue')
            legend.append('Target')
        if plot_raw and len(fr.raw):
            ax.plot(fr.frequency, fr.raw, linewidth=0.5, color='grey')
            legend.append('Raw')
        if plot_error and len(fr.error):
            ax.plot(fr.frequency, fr.error, linewidth=0.5, color='pink')
            legend.append('Error')
        if plot_smoothed and len(fr.smoothed):
            ax.plot(fr.frequency, fr.smoothed, linewidth=1, color='black')
            legend.append('Raw Smoothed')
        if plot_error_smoothed and len(fr.error_smoothed):
            ax.plot(fr.frequency, fr.error_smoothed, linewidth=1, color='red')
            legend.append('Error Smoothed')
        if plot_equalization and len(fr.equalization):
            ax.plot(fr.frequency, fr.equalization, linewidth=1, color='darkgreen')
            legend.append('Equalization')
        if plot_equalized and len(fr.equalized_raw) and not len(fr.equalized_smoothed):
            ax.plot(fr.frequency, fr.equalized_raw, linewidth=1, color='magenta')
            legend.append('Equalized raw')
        if plot_equalized and len(fr.equalized_smoothed):
            ax.plot(fr.frequency, fr.equalized_smoothed, linewidth=1, color='blue')
            legend.append('Equalized smoothed')

        ax.legend(legend, fontsize=8)

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
            - Figure
            - Axes
        """
        if fig is None:
            fig, ax = plt.subplots()
        t, decay = self.decay()
        ax.plot(t / self.fs * 1000, 20 * np.log10(decay), linewidth=1)

        ax.set_ylim([-120, 10])
        ax.set_xlim([
            int(self.peak_index() / self.fs - 1) * 1000,
            int(self.tail_index() / self.fs + 1) * 1000
        ])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (dBr)')
        ax.grid(True, which='major')
        ax.set_title('Decay')
        if plot_file_path:
            fig.savefig(plot_file_path)
        return fig, ax

    def plot_waterfall(self, fig=None, ax=None):
        """"""
        if fig is None:
            fig, ax = plt.subplots()

        z_min = -100

        # Window
        nfft = min(int(self.fs * 0.3), int(len(self.data) / 10))  # 300 ms or 1/10 of ir length
        noverlap = int(nfft * 0.9)  # 90% overlap
        ascend_ms = 10  # 10 ms ascending window
        ascend = int(ascend_ms / 1000 * self.fs)
        plateu = int((nfft - ascend) * 3 / 4)  # 75%
        descend = nfft - ascend - plateu  # 25%
        window = np.concatenate([
            hann(ascend * 2)[:ascend],
            np.ones(plateu),
            hann(descend * 2)[descend:]
        ])

        # Crop from 10ms before peak to start of tail
        peak_ind = self.peak_index()
        tail_ind = self.tail_index()
        start = max(int(peak_ind - self.fs * 0.01), 0)
        stop = min(tail_ind + nfft, len(self.data))
        data = self.data[start:stop]

        # Get spectrogram data
        spectrum, freqs, t = specgram(data, Fs=self.fs, NFFT=nfft, noverlap=noverlap, mode='magnitude', window=window)

        # Remove 0 Hz component
        spectrum = spectrum[1:, :]
        freqs = freqs[1:]

        # Interpolate to logaritmic frequency scale
        f_max = self.fs / 2
        f_min = 10
        step = 1.1
        f = np.array([f_min * step ** i for i in range(int(np.log(f_max / f_min) / np.log(step)))])
        log_f_spec = np.ones((len(f), spectrum.shape[1]))
        for i in range(spectrum.shape[1]):
            interpolator = InterpolatedUnivariateSpline(np.log10(freqs), spectrum[:, i], k=1)
            log_f_spec[:, i] = interpolator(np.log10(f))
        z = log_f_spec
        f = np.log10(f)

        # Normalize and turn to dB scale
        z /= np.max(z)
        z = np.clip(z, 10**(z_min/20), np.max(z))
        z = 20 * np.log10(z)

        # Smoothen
        z = uniform_filter(z, size=3, mode='constant')
        t, f = np.meshgrid(t, f)

        # Smoothing creates "walls", remove them
        t = t[1:-1, :-1] * 1000  # Milliseconds
        f = f[1:-1, :-1]
        z = z[1:-1, :-1]

        # Surface plot
        vmin = -100
        ax.plot_surface(t, f, z, cmap='magma', antialiased=True, vmin=z_min, vmax=0)

        # Z axis
        ax.set_zlim([z_min, 0])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # X axis
        ax.set_xlim([0, None])
        ax.set_xlabel('Time (ms)')

        # Y axis
        ax.set_ylim(np.log10([20, 20000]))
        ax.set_ylabel('Frequency (Hz)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{10 ** x:.0f}'))

        # Orient
        ax.view_init(30, 30)

        return fig, ax
