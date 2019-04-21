# -*- coding: utf-8 -*-

import os
import numpy as np
from pydub import AudioSegment


class Track:
    def __init__(self, name, data, fs):
        self.name = name
        self.data = data
        self.fs = fs

    @classmethod
    def read_wav(cls, file_path):
        # Using AudioSegment because SciPy can't read 24-bit WAVs
        seg = AudioSegment.from_wav(file_path)
        if len(seg.split_to_mono()) > 1:
            raise TypeError('WAV file has more than one track')
        data = np.array(seg.get_array_of_samples())

        dtype = data.dtype
        data = data.astype('float64')
        if dtype == 'int32':
            data /= 2.0 ** 31
        elif dtype == 'int16':
            data /= 2.0 ** 15
        elif dtype == 'uint8':
            data /= 2.0 ** 8
            data *= 2.0
            data -= 1.0

        dir_path, file_name = os.path.split(os.path.abspath(file_path))
        return cls(name=file_name[:-4], data=data, fs=seg.frame_rate)

    def __len__(self):
        return len(self.data)

    def tail(self):
        return None

    def amplify_db(self, gain_db):
        self.data *= 10**(gain_db / 20)

    def amplify(self, gain):
        self.data *= gain

    def absmax(self):
        return np.max(np.abs(self.data))

    def zeropad(self, new_size):
        self.data = np.concatenate([self.data, np.zeros(new_size - len(self))])


class TrackGroup:
    def __init__(self):
        self.fs = None
        self.tracks = []

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, i):
        return self.tracks[i]

    def add(self, sweep):
        if len(self.tracks):
            if sweep.fs != self.fs:
                raise ValueError('New Track has different sampling rate.')
        else:
            self.fs = sweep.fs
        self.tracks.append(sweep)

    def normalize(self, target_db=0.0):
        gain = np.max([s.absmax() for s in self.tracks]) * 10 ** (target_db / 20)
        for s in self.tracks:
            s.amplify(gain)

    def write_wav(self, file_path):
        if len(set([len(x) for x in self.tracks])) > 1:
            raise ValueError('Different length tracks cannot be stacked together in a single WAV file.')
        tracks = []
        for sweep in self.tracks:
            tracks.append(AudioSegment(
                np.multiply(sweep.data, 2 ** 31).astype('int32').tobytes(),
                frame_rate=self.fs,
                sample_width=4,
                channels=1
            ))
        seg = AudioSegment.from_mono_audiosegments(*tracks)
        seg.export(file_path, format='wav')