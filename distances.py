# -*- coding: utf-8 -*-

import numpy as np


def main():
    head_breadth = 0.16  # meters
    speaker_distance = 2  # meters
    speaker_angle = 60  # degrees
    sound_velocity = 343  # meters per second
    distance = np.sqrt(speaker_distance**2 + (head_breadth / 2)**2 - 2 * speaker_distance * (head_breadth / 2) * np.cos(speaker_angle / 180 * np.pi))
    delay = (speaker_distance - distance) / sound_velocity
    spl = -3.0 * np.log(distance / speaker_distance) / np.log(2)
    print(f'{distance:.2f} m, {delay*1000:.4f} ms, {spl:.4f} dB')


if __name__ == '__main__':
    main()
