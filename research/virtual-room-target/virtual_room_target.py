# -*- coding: utf-8 -*-

import os
import sys
import matplotlib.pyplot as plt
from autoeq.frequency_response import FrequencyResponse
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from utils import optimize_png_size

DIR_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


def main():
    # Flat loudspeaker in room respones
    flat_in_room = FrequencyResponse.read_from_csv(os.path.join(DIR_PATH, 'harman-flat-loudspeaker-in-room.csv'))
    flat_in_room.interpolate(f_step=1.01, f_min=10, f_max=20000)

    # Harman room target
    room_target = FrequencyResponse.read_from_csv(os.path.join(DIR_PATH, 'harman-room-target-original.csv'))
    room_target.interpolate(f_step=1.01, f_min=10, f_max=20000)
    room_target.center()
    room_target.smoothen_fractional_octave(window_size=1/3)
    room_target.raw = room_target.smoothed
    room_target.smoothed = []
    # Drob infra bass
    drop = room_target._sigmoid(f_lower=12, f_upper=24, a_normal=-70, a_treble=0)
    room_target.raw += drop

    # Harman 2018 over-ear headphone target
    over_ear = FrequencyResponse.read_from_csv(os.path.join(DIR_PATH, 'harman-over-ear-2018-without-bass.csv'))
    over_ear.interpolate(f_step=1.01, f_min=10, f_max=20000)
    over_ear.compensate(flat_in_room)
    over_ear.smoothen_fractional_octave(window_size=1/3)
    over_ear.raw = over_ear.smoothed.copy()
    over_ear.smoothed = []
    over_ear.error = over_ear.error_smoothed.copy()
    over_ear.error_smoothed = []

    # Virtual room target is Harman room target and the difference between Harman flat speaker in room and over-ear
    virtual_room_target = room_target.copy()
    virtual_room_target.raw += over_ear.error

    # Save room targets
    room_target.write_to_csv(os.path.join(DIR_PATH, 'harman-room-target.csv'))
    virtual_room_target.write_to_csv(os.path.join(DIR_PATH, 'virtual-room-target.csv'))

    # Plot
    fig, ax = over_ear.plot_graph(show=False)
    room_target.plot_graph(fig=fig, ax=ax, show=False, color='#1f77b4')
    virtual_room_target.plot_graph(fig=fig, ax=ax, show=False, color='#680fb9')
    plt.legend(['Harman flat loudspeaker in room', 'Harman over-ear 2018', 'Difference', 'Harman room target',
                'Virtual room target'])
    plt.xlim([10, 20000])
    plt.ylim([-65, 15])
    plt.title('Virtual Room Target')

    # Save figure
    figure_path = os.path.join(DIR_PATH, 'Results.png')
    fig.savefig(figure_path)
    optimize_png_size(figure_path)

    plt.show()


if __name__ == '__main__':
    main()
