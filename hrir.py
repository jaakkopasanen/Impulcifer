# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
from impulse_response import ImpulseResponse
from utils import read_wav, write_wav, magnitude_response
from constants import SPEAKER_NAMES, SPEAKER_DELAYS, IR_ORDER


class HRIR:
    def __init__(self, estimator):
        self.estimator = estimator
        self.fs = self.estimator.fs
        self.irs = dict()

    def open_recording(self, file_path, speakers, silence_length=2.0):
        """Open combined recording and splits it into separate speaker-ear pairs.

        Args:
            file_path: Path to recording file.
            speakers: Sequence of recorded speakers.
            silence_length: Length of silence used during recording in seconds.

        Returns:
            None
        """
        if self.fs != self.estimator.fs:
            raise ValueError('Refusing to open recording because HRIR\'s sampling rate doesn\'t match impulse response '
                             'estimator\'s sampling rate.')

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
            for j, column in enumerate(columns):
                n = int(i // 2 * len(columns) + j)
                speaker = speakers[n]
                if speaker not in SPEAKER_NAMES:
                    # Skip non-standard speakers. Useful for skipping the other sweep in center channel recording.
                    continue
                left = column[i, :]  # Left ear of current speaker
                right = column[i + 1, :]  # Right ear of current speaker
                if speaker not in self.irs:
                    self.irs[speaker] = dict()
                self.irs[speaker]['left'] = ImpulseResponse(self.estimator.estimate(left), self.fs, left)
                self.irs[speaker]['right'] = ImpulseResponse(self.estimator.estimate(right), self.fs, right)
            i += 2

    def write_wav(self, file_path, track_order=None, bit_depth=32):
        """Writes impulse responses to a WAV file

        Args:
            file_path: Path to output WAV file
            track_order: List of speaker-side names for the order of impulse responses in the output file
            bit_depth: Number of bits per sample. 16, 24 or 32

        Returns:
            None
        """
        # Duplicate speaker names as left and right side impulse response names
        if track_order is None:
            track_order = IR_ORDER

        # Add all impulse responses to a list and save channel names
        irs = []
        ir_order = []
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                irs.append(ir.data)
                ir_order.append(f'{speaker}-{side}')

        # Add silent tracks
        for ch in track_order:
            if ch not in ir_order:
                irs.append(np.zeros(len(irs[0])))
                ir_order.append(ch)
        irs = np.vstack(irs)

        # Sort to output order
        irs = irs[[ir_order.index(ch) for ch in track_order], :]

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
            left.append(pair['left'].data)
            right.append(pair['right'].data)
        left = np.sum(np.vstack(left), axis=0)
        right = np.sum(np.vstack(right), axis=0)
        # Calculate magnitude responses
        f_l, mr_l = magnitude_response(left, self.fs)
        f_r, mr_r = magnitude_response(right, self.fs)
        # Maximum absolute gain from both sides
        gain = np.max(np.vstack([mr_l, mr_r])) * -1 + target_db
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                ir.data *= 10 ** (gain / 20)

    def crop_heads(self, head_ms=1):
        """Crops heads of impulse responses

        Args:
            head_ms: Milliseconds of head room in the beginning before impulse response max which will not be cropped

        Returns:
            None
        """
        if self.fs != self.estimator.fs:
            raise ValueError('Refusing to crop heads because HRIR sampling rate doesn\'t match impulse response '
                             'estimator\'s sampling rate.')

        for speaker, pair in self.irs.items():
            # Peaks
            peak_left = pair['left'].peak_index()
            peak_right = pair['right'].peak_index()
            itd = np.abs(peak_left - peak_right) / self.fs

            # Speaker channel delay
            head = head_ms * self.fs // 1000
            delay = int(np.round(SPEAKER_DELAYS[speaker] / 1000 * self.fs)) + head  # Channel delay in samples

            if peak_left < peak_right:
                # Delay to left ear is smaller, this is must left side speaker
                if speaker[1] == 'R':
                    # Speaker name indicates this is right side speaker but delay to left ear is smaller than to right.
                    # There is something wrong with the measurement
                    warnings.warn(f'Warning: {speaker} measurement has lower delay to left ear than to right ear. '
                                  f'{speaker} should be at the right side of the head so the sound should arrive first '
                                  f'in the right ear. This is usually a problem with the measurement process or the '
                                  f'speaker order given is not correct. Detected delay difference is '
                                  f'{itd * 1000:.4f} milliseconds.')
                # Crop out silence from the beginning, only required channel delay remains
                # Secondary ear has additional delay for inter aural time difference
                pair['left'].data = pair['left'].data[peak_left - delay:]
                pair['right'].data = pair['right'].data[peak_left - delay:]
            else:
                # Delay to right ear is smaller, this is must right side speaker
                if speaker[1] == 'L':
                    # Speaker name indicates this is left side speaker but delay to right ear is smaller than to left.
                    # There si something wrong with the measurement
                    warnings.warn(f'Warning: {speaker} measurement has lower delay to right ear than to left ear. '
                                  f'{speaker} should be at the left side of the head so the sound should arrive first '
                                  f'in the left ear. This is usually a problem with the measurement process or the '
                                  f'speaker order given is not correct. Detected delay difference is '
                                  f'{itd * 1000:.4f} milliseconds.')
                # Crop out silence from the beginning, only required channel delay remains
                # Secondary ear has additional delay for inter aural time difference
                pair['right'].data = pair['right'].data[peak_right - delay:]
                pair['left'].data = pair['left'].data[peak_right - delay:]

            # peak_left = pair['left'].peak_index()
            # peak_right = pair['right'].peak_index()
            # window_start = max(0, min(peak_left, peak_right) - 50)
            # window_end = min(len(pair['left'].data), max(peak_left, peak_right) + 50)
            # plt.plot(
            #     np.arange(window_start, window_end)/self.fs*1000,
            #     pair['left'].data[window_start:window_end],
            #     color='#2f29ce'
            # )
            # plt.plot(
            #     np.arange(window_start, window_end)/self.fs*1000,
            #     pair['right'].data[window_start:window_end],
            #     color='#e5312f'
            # )
            # plt.plot(peak_left/self.fs*1000, pair['left'].data[peak_left], 'o', color='black')
            # plt.plot(peak_right/self.fs*1000, pair['right'].data[peak_right], 'o', color='black')
            # plt.legend(['Left', 'Right'])
            # plt.suptitle(speaker)
            # plt.title(f'ITD = {(peak_right - peak_left)/self.fs*1000:.4f}ms')
            # plt.grid()
            # plt.show()

            # Make sure impulse response starts from silence
            window = signal.hanning(head * 2)[:head]
            pair['left'].data[:head] *= window
            pair['right'].data[:head] *= window

    def crop_tails(self):
        """Crops out tails after every impulse response has decayed to noise floor."""
        if self.fs != self.estimator.fs:
            raise ValueError('Refusing to crop tails because HRIR\'s sampling rate doesn\'t match impulse response '
                             'estimator\'s sampling rate.')
        # Find indices after which there is only noise in each track
        tail_indices = []
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                tail_indices.append(ir.tail_index())

        # Crop all tracks by last tail index
        seconds_per_octave = len(self.estimator) / self.fs / self.estimator.n_octaves
        fade_out = 2 * int(self.fs * seconds_per_octave * (1 / 24))
        window = signal.hanning(fade_out)[fade_out // 2:]
        tail_ind = max(tail_indices)
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                ir.data = ir.data[:tail_ind]
                ir.data *= np.concatenate([np.ones(len(ir.data) - len(window)), window])

    def plot(self,
             dir_path=None,
             plot_recording=True,
             plot_ir=True,
             plot_fr=True,
             plot_decay=True,
             plot_spectrogram=True,
             plot_waterfall=True):
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
        lims = {name: {'xlim': [None, None], 'ylim': [None, None]} for name in ['recording', 'ir', 'fr', 'decay']}
        plots = {name: {side: {'fig': None, 'ax': None} for side in ['left', 'right']} for name in self.irs.keys()}
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                # Create figure and axises for the plots
                fig = plt.figure()
                ax = []
                for i in range(5):
                    ax.append(fig.add_subplot(2, 3, i + 1))
                ax.append(fig.add_subplot(2, 3, 6, projection='3d'))
                ax = np.vstack([ax[:3], ax[3:]])
                fig.set_size_inches(22, 10)
                fig.suptitle(f'{speaker}-{side}')
                plots[speaker][side]['fig'] = fig
                plots[speaker][side]['ax'] = ax
                if plot_recording:
                    ir.plot_recording(fig=fig, ax=ax[0, 0])
                    max_lims(lims['recording'], ax[0, 0])
                if plot_spectrogram:
                    ir.plot_spectrogram(fig=fig, ax=ax[1, 0])
                if plot_ir:
                    peak = ir.peak_index() / self.fs
                    ir.plot_ir(fig=fig, ax=ax[0, 1], start=peak - 0.01, end=peak + 0.09)
                    max_lims(lims['ir'], ax[0, 1])
                if plot_fr:
                    ir.plot_fr(fig=fig, ax=ax[1, 1])
                    max_lims(lims['fr'], ax[1, 1])
                if plot_decay:
                    ir.plot_decay(fig=fig, ax=ax[0, 2])
                    max_lims(lims['decay'], ax[0, 2])
                if plot_waterfall:
                    ir.plot_waterfall(fig=fig, ax=ax[1, 2])

        # Synchronize axis limits for easier comparison
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                if plot_recording:
                    plots[speaker][side]['ax'][0, 0].set_xlim(lims['recording']['xlim'])
                    plots[speaker][side]['ax'][0, 0].set_ylim(lims['recording']['ylim'])
                if plot_ir:
                    plots[speaker][side]['ax'][0, 1].set_xlim(lims['ir']['xlim'])
                    plots[speaker][side]['ax'][0, 1].set_ylim(lims['ir']['ylim'])
                if plot_fr:
                    plots[speaker][side]['ax'][1, 1].set_xlim(lims['fr']['xlim'])
                    plots[speaker][side]['ax'][1, 1].set_ylim(lims['fr']['ylim'])
                if plot_decay:
                    plots[speaker][side]['ax'][0, 2].set_xlim(lims['decay']['xlim'])
                    plots[speaker][side]['ax'][0, 2].set_ylim(lims['decay']['ylim'])

        # Show plots and write figures to files
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                if dir_path is not None:
                    file_path = os.path.join(dir_path, f'{speaker}-{side}.png')
                    plots[speaker][side]['fig'].savefig(file_path, bbox_inches='tight')
                    # Optimize file size
                    im = Image.open(file_path)
                    im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
                    im.save(file_path, optimize=True)

    def equalize(self, fir):
        """Equalizes all impulse responses with given FIR filters.

        First row of the fir matrix will be used for all left side impulse responses and the second row for all right
        side impulse responses.

        Args:
            fir: FIR filter as an array like. Must have same sample rate as this HRIR instance.

        Returns:
            None
        """
        if type(fir) == list:
            # Turn list (list|array|ImpulseResponse) into Numpy array
            if type(fir[0]) == np.ndarray:
                fir = np.vstack(fir)
            elif type(fir[0]) == list:
                fir = np.array(fir)
            elif type(fir[0]) == ImpulseResponse:
                if len(fir) > 1:
                    fir = np.vstack([fir[0].data, fir[1].data])
                else:
                    fir = fir[0].data.copy()

        if len(fir.shape) == 1 or fir.shape[0] == 1:
            # Single track in the WAV file, use it for both channels
            fir = np.tile(fir, (2, 1))

        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                ir.equalize(fir[0] if side == 'left' else fir[1])

    def resample(self, fs):
        """Resamples all impulse response to the given sampling rate.

        Sets internal sampling rate to the new rate. This will disable file reading and cropping so this should be
        the last method called in the processing pipeline.

        Args:
            fs: New sampling rate in Hertz

        Returns:
            None
        """
        for speaker, pair in self.irs.items():
            for side, ir in pair.items():
                ir.resample(fs)
        self.fs = fs
