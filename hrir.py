# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import kaiser, hanning, correlate
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
                speaker = speakers[i // 2 + j]
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
        for speaker, pair in self.irs.items():
            # Peaks
            peak_left = pair['left'].peak_index()
            peak_right = pair['right'].peak_index()

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
                pair['left'].data = pair['left'].data[peak_left - delay:]
                pair['right'].data = pair['right'].data[peak_left - delay:]
            else:
                # Delay to right ear is smaller, this is must right side speaker
                if speaker[1] == 'L':
                    # Speaker name indicates this is left side speaker but delay to right ear is smaller than to left.
                    # There si something wrong with the measurement
                    raise ValueError(speaker + ' impulse response has lower delay to right ear than to left.')
                # Crop out silence from the beginning, only required channel delay remains
                # Secondary ear has additional delay for inter aural time difference
                pair['right'].data = pair['right'].data[peak_right - delay:]
                pair['left'].data = pair['left'].data[peak_right - delay:]

            # Make sure impulse response starts from silence
            window = hanning(head * 2)[:head]
            pair['left'].data[:head] *= window
            pair['right'].data[:head] *= window

    def crop_tails(self):
        """Crops out tails after every impulse response has decayed to noise floor."""
        # Find indices after which there is only noise in each track
        tail_indices = []
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                tail_indices.append(ir.tail_index())

        # Crop all tracks by last tail index
        seconds_per_octave = len(self.estimator) / self.fs / self.estimator.n_octaves
        fade_out = 2 * int(self.fs * seconds_per_octave * (1 / 24))
        window = hanning(fade_out)[fade_out // 2:]
        tail_ind = max(tail_indices)
        for speaker, pair in self.irs.items():
            for ir in pair.values():
                ir.data = ir.data[:tail_ind]
                ir.data *= np.concatenate([np.ones(len(ir.data) - len(window)), window])

    def plot(self, dir_path=None):
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
                fig.suptitle(f'{speaker}-{side}')
                plots[speaker][side]['fig'] = fig
                plots[speaker][side]['ax'] = ax
                # Impulse response
                ir.plot_ir(fig=fig, ax=ax[0, 0])
                max_lims(lims['ir'], ax[0, 0])
                # Frequency response
                ir.plot_fr(fig=fig, ax=ax[0, 1])
                max_lims(lims['fr'], ax[0, 1])
                # Decay graph
                ir.plot_decay(fig=fig, ax=ax[1, 0], window_size_ms=0.1)
                max_lims(lims['decay'], ax[1, 0])
                # Spectrogram
                ir.plot_spectrogram(fig=fig, ax=ax[1, 1])

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
                    file_path = os.path.join(dir_path, f'{speaker}-{side}.png')
                    plots[speaker][side]['fig'].savefig(file_path, dpi=480, bbox_inches='tight')
                    # Optimize file size
                    im = Image.open(file_path)
                    im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
                    im.save(file_path, optimize=True)
