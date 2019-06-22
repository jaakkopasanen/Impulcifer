# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
from scipy.signal import kaiser, hanning
from PIL import Image
from autoeq.frequency_response import FrequencyResponse
from utils import read_wav, write_wav, magnitude_response
from constants import SPEAKER_NAMES, SPEAKER_DELAYS


class ImpulseResponse:
    def __init__(self, recording, estimator):
        self.recording = recording
        self.estimator = estimator
        self.fs = self.estimator.fs
        self.ir = estimator.estimater(recording)

    def plot_spectrogram(self, fig=None, ax=None, show_plot=False, plot_file_path=None):
        """Plots spectrogram for a logarithmic sine sweep recording.

        Args:
            ax: Axis
            show_plot: Show plot live?
            plot_file_path: Path to a file for saving the plot

        Returns:
            None
        """
        if len(np.nonzero(self.recording)[0]) == 0:
            return

        if fig is None:
            fig, ax = plt.subplots()
        ax.specgram(self.recording, Fs=self.fs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')

        if plot_file_path:
            fig.savefig(plot_file_path)
        if show_plot:
            plt.show(fig)

    def decay(self, window_size_ms=1):
        """Decay graph with RMS values for each window.

        Args:
            window_size_ms: RMS window size in milliseconds.

        Returns:
            RMS widows as Numpy array
        """
        # Sliding window RMS
        window_size = self.fs // 1000 * window_size_ms

        # RMS windows
        n = len(self.ir) // window_size
        windows = np.vstack(np.split(self.ir[:n * window_size], n))
        rms = np.sqrt(np.mean(np.square(windows), axis=1))

        # Prevent division by zero in log10
        rms[rms == 0.0] = 1e-9

        # Smoothen data
        smoothed = 20 * np.log10(rms)
        for _ in range(200):
            smoothed = signal.savgol_filter(smoothed, 11, 1)

        return smoothed

    def plot_decay(self, window_size_ms=1, fig=None, ax=None, show_plot=False, plot_file_path=None):
        """Plots decay graph.

        Args:
            window_size_ms: RMS window size in milliseconds.
            fig: Figure instance. New will be created if None is passed.
            ax: Axis instance. New will be created if None is passed to fig.
            show_plot: Show plot live.
            plot_file_path: Save plot figure to a file.

        Returns:
            None
        """
        rms = self.decay(window_size_ms=window_size_ms)
        if show_plot or plot_file_path:
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

    def plot_ir(self, fig=None, ax=None, max_time=None, show_plot=False, plot_file_path=None):
        """Plots impulse response wave form.

        Args:
            fig: Figure instance
            ax: Axis instance
            max_time: Maximum time in seconds for cropping the tail.
            show_plot: Show plot live?
            plot_file_path: Path to a file for saving the plot

        Returns:
            None
        """
        if max_time is None:
            max_time = len(self.ir) / self.fs
        ir = self.ir[:int(max_time * self.fs)]

        if fig is None:
            fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(ir) / self.fs * 1000, 1000 / self.fs), ir, linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.grid(True)
        ax.set_title('Impulse response {ms} ms'.format(ms=int(max_time * 1000)))

        if plot_file_path:
            fig.savefig(plot_file_path)
        if show_plot:
            plt.show()

    def plot_fr(self, fig=None, ax=None, show_plot=False, plot_file_path=None):
        """Plots frequency response."""
        f, m = magnitude_response(self.ir, self.fs)
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
        ax.plot(fr.frequency, fr.raw, linewidth=0.5)
        ax.plot(fr.frequency, fr.smoothed, linewidth=1)
        ax.legend(['Raw', 'Smoothed'], fontsize=8)

        if plot_file_path:
            fig.savefig(plot_file_path)
        if show_plot:
            plt.show()


class HRIR:
    def __init__(self, estimator):
        self.estimator = estimator
        self.fs = self.estimator.fs
        self.irs = {speaker: {'left': None, 'right': None} for speaker in SPEAKER_NAMES}

    def open_recording(self, file_path, speakers, silence_length=2.0):
        """Open combined recording and splits it into separate speaker-ear pairs.

        Args:
            file_path: Path to recording file.
            speakers: Sequence of recorded speakers.
            silence_length: Length of silence used during recording in seconds.

        Returns:
            None
        """
        fs, recording = read_wav(file_path)
        if fs != self.fs:
            raise ValueError('Sampling rate of recording must match sampling rate of test signal.')

        if silence_length * self.fs != int(silence_length * self.fs):
            raise ValueError('Silence length must produce full samples with given sampling rate.')
        silence_length = int(silence_length * self.fs)

        # Number of speakers in each track
        n_columns = round(len(speakers) / (recording.shape[0] // 2))

        # Crop out initial silence
        recording = recording[:, silence_length:]

        # Split sections in time to columns
        columns = []
        column_size = silence_length + len(self.estimator)
        for i in range(n_columns):
            columns.append(recording[:, i * column_size:(i + 1) * column_size])

        # Split each track by columns
        i = 0
        while i < recording.shape[0]:
            for column in columns:
                speaker = speakers[i // 2]
                left = column[i, :]  # Left ear of current speaker
                right = column[i + 1, :]  # Right ear of current speaker
                self.irs[speaker]['left'] = ImpulseResponse(left, self.estimator)
                self.irs[speaker]['right'] = ImpulseResponse(right, self.estimator)
            i += 2

    def write_wav(self, file_path, speakers=SPEAKER_NAMES, bit_depth=32):
        """Writes impulse responses to a WAV file

        Args:
            file_path: Path to output WAV file
            speakers: List of speaker names for the order of impulse responses in the output file
            bit_depth: Number of bits per sample. 16, 24 or 32

        Returns:
            None
        """
        # Duplicate speaker names as left and right side impulse response names
        output_ir_order = []
        for speaker in speakers:
            output_ir_order.append('{speaker}-{side}'.format(speaker=speaker, side='left'))
            output_ir_order.append('{speaker}-{side}'.format(speaker=speaker, side='right'))

        # Add all impulse responses to a list and save channel names
        irs = []
        ir_order = []
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                irs.append(ir.ir)
                ir_order.append('{speaker}-{side}'.format(speaker=speaker, side=side))

        # Add silent tracks
        for ch in output_ir_order:
            if ch not in ir_order:
                irs.append(np.zeros(len(irs[0])))
                ir_order.append(ch)
        irs = np.vstack(irs)

        # Sort to output order
        irs = irs[[ir_order.index(ch) for ch in output_ir_order], :]

        # Write to file
        write_wav(file_path, self.fs, irs, bit_depth=bit_depth)

    def normalize(self, target_db=-0.1):
        """Normalizes output gain to target.

        Args:
            target_db: Target gain in dB

        Returns:

        """
        # Stack and sum all left and right ear impulse responses separately
        left = []
        right = []
        for speaker, pair in self.irs.items():
            left.append(pair['left'].ir)
            right.append(pair['right'].ir)
        left = np.sum(np.vstack(left), axis=0)
        right = np.sum(np.vstack(right), axis=0)
        # Calculate magnitude responses
        f_l, mr_l = magnitude_response(left, self.fs)
        f_r, mr_r = magnitude_response(right, self.fs)
        # Maximum absolute gain from both sides
        gain = np.max(np.abs(np.vstack(mr_l, mr_r))) * -1
        print(gain)
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                ir.ir *= 10 ** ((gain + target_db) / 20)

    def crop_heads(self, head_ms=1):
        """Crops heads of impulse responses

        Args:
            head_ms: Milliseconds of head room in the beginning before impulse response max which will not be cropped

        Returns:
            None
        """
        for speaker, pair in self.irs.items():
            left = pair['left']
            right = pair['right']

            # Peaks
            peak_left, _ = signal.find_peaks(left / np.max(left), height=0.1)
            peak_left = peak_left[0]
            peak_right, _ = signal.find_peaks(right / np.max(right), height=0.1)
            peak_right = peak_right[0]
            # Inter aural time difference (in samples)
            itd = np.abs(peak_left - peak_right)

            # Speaker channel delay
            head = head_ms * self.fs // 1000
            delay = int(np.round(SPEAKER_DELAYS[speaker] / 1000 * self.fs)) + head  # Channel delay in samples

            if peak_left < peak_right:
                # Delay to left ear is smaller, this is must left side speaker
                if speaker[1] == 'R':
                    # Speaker name indicates this is right side speaker but delay to left ear is smaller than to right.
                    # There is something wrong with the measurement
                    raise ValueError(speaker + ' impulse response has lower delay to left ear than to right.')
                # Crop out silence from the beginning, only required channel delay remains
                # Secondary ear has additional delay for inter aural time difference
                left = left[peak_left - delay:]
                right = right[peak_right - (delay + itd):]
            else:
                # Delay to right ear is smaller, this is must right side speaker
                if speaker[1] == 'L':
                    # Speaker name indicates this is left side speaker but delay to right ear is smaller than to left.
                    # There si something wrong with the measurement
                    raise ValueError(speaker + ' impulse response has lower delay to right ear than to left.')
                # Crop out silence from the beginning, only required channel delay remains
                # Secondary ear has additional delay for inter aural time difference
                left = left[peak_left - (delay + itd):]
                right = right[peak_right - delay:]

            # Make sure impulse response starts from silence
            window = hanning(head * 2)[:head]
            left[:head] *= window
            right[:head] *= window

    def crop_tails(self):
        """Crops out tails after every impulse response has decayed to noise floor."""
        # Find indices after which there is only noise in each track
        tail_indices = []
        for speaker, pair in self.irs.items():
            for ir in pair:
                tail_indices.append(ir.tail_index())

        # Crop all tracks by last tail index
        tail_ind = max(tail_indices)
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                ir.ir = ir.ir[:tail_ind]

    def plot(self, show_plot=False, dir_path=None):
        """Plots all impulse responses."""
        def max_lims(_lims, _ax):
            if _lims['xlim'][0] is None or _ax.get_xlim()[0] < _lims['xlim'][0]:
                _lims['xlim'][0] = _ax.get_xlim()[0]
            if _lims['xlim'][1] is None or _ax.get_xlim()[1] > _lims['xlim'][1]:
                _lims['xlim'][1] = _ax.get_xlim()[1]
            if _lims['ylim'][0] is None or _ax. get_ylim()[0] < _lims['ylim'][0]:
                _lims['ylim'][0] = _ax.get_ylim()[0]
            if _lims['ylim'][1] is None or _ax. get_ylim()[1] > _lims['ylim'][1]:
                _lims['ylim'][1] = _ax.get_ylim()[1]

        # Plot and save max limits
        lims = {name: {'xlim': [None, None], 'ylim': [None, None]} for name in ['ir', 'fr', 'decay']}
        plots = {name: {side: {'fig': None, 'ax': None} for side in ['left', 'right']} for name in self.irs.keys()}
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                # Create figure and axises for the plots
                fig, ax = plt.subplots(2, 2)
                fig.set_size_inches(15, 10)
                plots[speaker][side]['fig'] = fig
                plots[speaker][side]['ax'] = ax
                # Impulse response
                ir.plot_ir(show_plot=False, fig=fig, ax=ax[0, 0])
                max_lims(lims['ir'], ax[0, 0])
                # Frequency response
                ir.plot_fr(show_plot=False, fig=fig, ax=ax[0, 1])
                max_lims(lims['fs'], ax[0, 1])
                # Decay graph
                ir.plot_decay(show_plot=False, fig=fig, ax=ax[1, 0])
                max_lims(lims['decay'], ax[1, 0])
                # Spectrogram
                ir.plot_spectrogram(show_plot=show_plot, fig=fig, ax=ax[1, 1])

        # Synchronize axis limits for easier comparison
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                # Impulse response
                plots[speaker][side]['ax'][0, 0].set_xlim(lims['ir']['xlim'])
                plots[speaker][side]['ax'][0, 0].set_ylim(lims['ir']['ylim'])
                # Frequency response
                plots[speaker][side]['ax'][0, 1].set_xlim(lims['fr']['xlim'])
                plots[speaker][side]['ax'][0, 1].set_ylim(lims['fr']['ylim'])
                # Decay graph
                plots[speaker][side]['ax'][1, 0].set_xlim(lims['decay']['xlim'])
                plots[speaker][side]['ax'][1, 0].set_ylim(lims['decay']['ylim'])

        # Show plots and write figures to files
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                if dir_path is not None:
                    file_path = os.path.join(dir_path, '{speaker}-{side}.png'.format(speaker=speaker, side=side))
                    plots[speaker][side]['fig'].savefig(file_path, dpi=480, bbox_inches='tight')
                    # Optimize file size
                    im = Image.open(file_path)
                    im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
                    im.save(file_path, optimize=True)
                if show_plot:
                    plt.show(plots[speaker][side]['fig'])
