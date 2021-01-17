# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from autoeq.frequency_response import FrequencyResponse
from impulse_response import ImpulseResponse
from hrir import HRIR
from utils import sync_axes, save_fig_as_png, read_wav, get_ylim, config_fr_axis
from impulcifer_constants import SPEAKER_NAMES, SPEAKER_LIST_PATTERN, IR_ROOM_SPL, COLORS


def room_correction(
        estimator,
        dir_path,
        target=None,
        mic_calibration=None,
        fr_combination_method='average',
        specific_limit=20000,
        generic_limit=1000,
        plot=False):
    """Corrects room acoustics

    Args:
        estimator: ImpulseResponseEstimator
        dir_path: Path to directory
        target: Path to room target response CSV file
        mic_calibration: Path to room measurement microphone calibration file
        fr_combination_method: Method for combining generic room measurment frequency responses. "average" or
                               "conservative"
        specific_limit: Upper limit in Hertz for equalization of specific room eq. 0 disables limit.
        generic_limit: Upper limit in Hertz for equalization of generic room eq. 0 disables limit.
        plot: Plot graphs?

    Returns:
        - Room Impulse Responses as HRIR or None
        - Equalization frequency responses as dict of dicts (similar to HRIR) or None
    """
    # Open files
    target = open_room_target(estimator, dir_path, target=target)
    mic_calibration = open_mic_calibration(estimator, dir_path, mic_calibration=mic_calibration)
    rir = open_room_measurements(estimator, dir_path)
    missing = [ch for ch in SPEAKER_NAMES if ch not in rir.irs]
    room_fr = open_generic_room_measurement(
        estimator,
        dir_path,
        mic_calibration,
        target,
        method=fr_combination_method,
        limit=generic_limit,
        plot=plot
    )

    if not len(rir.irs) and room_fr is None:
        # No room recording files found
        return None, None

    frs = dict()
    fr_axes = []
    figs = None
    if len(rir.irs):
        # Crop heads and tails from room impulse responses
        for speaker, pair in rir.irs.items():
            for side, ir in pair.items():
                ir.crop_head()
        rir.crop_tails()
        rir.write_wav(os.path.join(dir_path, 'room-responses.wav'))

        if plot:
            # Plot all but frequency response
            plot_dir = os.path.join(dir_path, 'plots', 'room')
            os.makedirs(plot_dir, exist_ok=True)
            figs = rir.plot(plot_fr=False, close_plots=False)

        # Create equalization frequency responses
        reference_gain = None
        for speaker, pair in rir.irs.items():
            frs[speaker] = dict()
            for side, ir in pair.items():
                # Create frequency response
                fr = ir.frequency_response()

                if mic_calibration is not None:
                    # Calibrate frequency response
                    fr.raw -= mic_calibration.raw

                # Sync gains
                if reference_gain is None:
                    reference_gain = fr.center([100, 10000])  # Shifted (up) by this many dB
                else:
                    fr.raw += reference_gain

                # Adjust target level with the (negative) gain caused by speaker-ear distance in reverberant room
                target_adjusted = target.copy()
                target_adjusted.raw += IR_ROOM_SPL[speaker][side]
                # Compensate with the adjusted room target
                fr.compensate(target_adjusted, min_mean_error=False)

                # Zero error above limit
                if specific_limit > 0:
                    start = np.argmax(fr.frequency > specific_limit / 2)
                    end = np.argmax(fr.frequency > specific_limit)
                    mask = np.concatenate([
                        np.ones(start if start > 0 else 0),
                        signal.windows.hann(end - start),
                        np.zeros(len(fr.frequency) - end)
                    ])
                    fr.error *= mask

                # Add frequency response
                frs[speaker][side] = fr

                if plot:
                    file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                    fr = fr.copy()
                    fr.smoothen_fractional_octave(window_size=1/3, treble_window_size=1/3)
                    _, fr_ax = ir.plot_fr(
                        fr=fr,
                        fig=figs[speaker][side],
                        ax=figs[speaker][side].get_axes()[4],
                        plot_raw=False,
                        plot_error=False,
                        plot_file_path=file_path,
                        fix_ylim=True
                    )
                    fr_axes.append(fr_ax)

    if len(missing) > 0 and room_fr is not None:
        # Use generic measurement for speakers that don't have specific measurements
        for speaker in missing:
            frs[speaker] = {'left': room_fr.copy(), 'right': room_fr.copy()}

    if plot and figs is not None:
        room_plots_dir = os.path.join(dir_path, 'plots', 'room')
        os.makedirs(room_plots_dir, exist_ok=True)
        # Sync FR plot axes
        sync_axes(fr_axes)
        # Save specific fR figures
        for speaker, pair in figs.items():
            for side, fig in pair.items():
                save_fig_as_png(os.path.join(room_plots_dir, f'{speaker}-{side}.png'), fig)
                plt.close(fig)

    return rir, frs


def open_room_measurements(estimator, dir_path):
    """Opens speaker-ear specific room measurements.

    Args:
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to directory

    Returns:
        HRIR instance with the room measurements
    """
    # Read room measurement files
    rir = HRIR(estimator)
    # room-BL,SL.wav, room-left-FL,FR.wav, room-right-FC.wav, etc...
    pattern = rf'^room-{SPEAKER_LIST_PATTERN}(-(left|right))?\.wav$'
    for i, file_name in enumerate([f for f in os.listdir(dir_path) if re.match(pattern, f)]):
        # Read the speaker names from the file name into a list
        speakers = re.search(SPEAKER_LIST_PATTERN, file_name)
        if speakers is not None:
            speakers = speakers[0].split(',')
        # Form absolute path
        file_path = os.path.join(dir_path, file_name)
        # Read side if present
        side = re.search(r'(left|right)', file_name)
        if side is not None:
            side = side[0]
        # Read file
        rir.open_recording(file_path, speakers, side=side)
    return rir


def open_generic_room_measurement(estimator,
                                  dir_path,
                                  mic_calibration,
                                  target,
                                  method='average',
                                  limit=1000,
                                  plot=False):
    """Opens generic room measurment file

    Args:
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to directory
        mic_calibration: Measurement microphone calibration FrequencyResponse
        target: Room response target FrequencyResponse
        method: Combination method. "average" or "conservative"
        limit: Upper limit in Hertz for equalization. Gain will ramp down to 0 dB in the octave leading to this.
               0 disables limit.
        plot: Plot frequency response?

    Returns:
        Generic room measurement FrequencyResponse
    """
    file_path = os.path.join(dir_path, 'room.wav')
    if not os.path.isfile(file_path):
        return None

    # Read the file
    fs, data = read_wav(file_path, expand=True)

    if fs != estimator.fs:
        raise ValueError(f'Sampling rate of "{file_path}" doesn\'t match!')

    # Average frequency responses of all tracks of the generic room measurement file
    irs = []
    for track in data:
        n_cols = int(round((len(track) / estimator.fs - 2) / (estimator.duration + 2)))
        for i in range(n_cols):
            # Starts at 2 seconds in the beginning plus previous sweeps and their tails
            start = int(2 * estimator.fs + i * (2 * estimator.fs + len(estimator)))
            # Ends at start plus one more (current) sweep
            end = int(start + 2 * estimator.fs + len(estimator))
            end = min(end, len(track))
            # Select current sweep
            sweep = track[start:end]
            # Deconvolve as impulse response
            ir = ImpulseResponse(estimator.estimate(sweep), estimator.fs, sweep)
            # Crop harmonic distortion from the head
            # Noise in the tail should not affect frequency response so it doesn't have to be cropped
            ir.crop_head(head_ms=1)
            irs.append(ir)

    # Frequency response for the generic room measurement
    room_fr = FrequencyResponse(
        name='generic_room',
        frequency=FrequencyResponse.generate_frequencies(f_min=10, f_max=estimator.fs / 2, f_step=1.01),
        raw=0, error=0, target=target.raw
    )

    # Calculate and stack errors
    raws = []
    errors = []
    for ir in irs:
        fr = ir.frequency_response()
        if mic_calibration is not None:
            fr.raw -= mic_calibration.raw
        fr.center([100, 10000])
        room_fr.raw += fr.raw
        raws.append(fr.copy())
        fr.compensate(target, min_mean_error=True)
        if method == 'conservative' and len(irs) > 1:
            fr.smoothen_fractional_octave(window_size=1/3, treble_window_size=1/3)
            errors.append(fr.error_smoothed)
        else:
            errors.append(fr.error)
    room_fr.raw /= len(irs)
    errors = np.vstack(errors)

    if errors.shape[0] > 1:
        # Combine errors
        if method == 'conservative':
            # Conservative error curve is zero everywhere else but on indexes where both have the same sign,
            # at these indexes the smaller absolute value is selected.
            # This ensures that no curve will be adjusted to the other side of zero
            mask = np.mean(errors > 0, axis=0)  # Average from boolean values per column
            positive = mask == 1  # Mask for columns with only positive values
            negative = mask == 0  # Mask for columns with only negative values
            # Minimum value for columns with only positive values
            room_fr.error[positive] = np.min(errors[:, positive], axis=0)
            # Maximum value for columns with only negative values (minimum absolute value)
            room_fr.error[negative] = np.max(errors[:, negative], axis=0)
            # Smoothen out kinks
            room_fr.smoothen_fractional_octave(window_size=1 / 6, treble_window_size=1 / 6)
            room_fr.error = room_fr.error_smoothed.copy()
        elif method == 'average':
            room_fr.error = np.mean(errors, axis=0)
            room_fr.smoothen_fractional_octave(window_size=1/3, treble_window_size=1/3)
        else:
            raise ValueError(
                f'Invalid value "{method}" for method. Supported values are "conservative" and "average"')
    else:
        room_fr.error = errors[0, :]
        room_fr.smoothen_fractional_octave(window_size=1 / 3, treble_window_size=1 / 3)

    if limit > 0:
        # Zero error above limit
        start = np.argmax(room_fr.frequency > limit / 2)
        end = np.argmax(room_fr.frequency > limit)
        mask = np.concatenate([
            np.ones(start if start > 0 else 0),
            signal.windows.hann(end - start),
            np.zeros(len(room_fr.frequency) - end)
        ])
        room_fr.error *= mask
        room_fr.error_smoothed *= mask

    if plot:
        # Create dir
        room_plots_dir = os.path.join(dir_path, 'plots', 'room')
        os.makedirs(room_plots_dir, exist_ok=True)

        # Create generic FR plot
        fr = room_fr.copy()
        fr.name = 'Generic room measurement'
        fr.raw = fr.smoothed.copy()
        fr.error = fr.error_smoothed.copy()

        # Create figure and axes
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 9)
        config_fr_axis(ax)
        ax.set_title('Generic room measurement')

        # Plot target, raw and error
        ax.plot(fr.frequency, fr.target, color=COLORS['lightpurple'], linewidth=5, label='Target')
        for raw in raws:
            raw.smoothen_fractional_octave(window_size=1/3, treble_window_size=1/3)
            ax.plot(raw.frequency, raw.smoothed, color='grey', linewidth=0.5)
        ax.plot(fr.frequency, fr.raw, color=COLORS['blue'], label='Raw smoothed')
        ax.plot(fr.frequency, fr.error, color=COLORS['red'], label='Error smoothed')
        ax.legend()

        # Set y limits
        sl = np.logical_and(fr.frequency >= 20, fr.frequency <= 20000)
        stack = np.vstack([
            fr.raw[sl],
            fr.error[sl],
            fr.target[sl]
        ])
        ax.set_ylim(get_ylim(stack, padding=0.1))

        # Save FR figure
        save_fig_as_png(os.path.join(room_plots_dir, 'room.png'), fig)
        plt.close(fig)

    return room_fr


def open_room_target(estimator, dir_path, target=None):
    """Opens room frequency response target file.

    Args:
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to directory
        target: Path to explicitly given (if any) room response target file

    Returns:
        Room response target FrequencyResponse
    """
    # Room target
    if target is None:
        target = os.path.join(dir_path, 'room-target.csv')
    if os.path.isfile(target):
        # File exists, create frequency response
        target = FrequencyResponse.read_from_csv(target)
        target.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)
        target.center()
    else:
        # No room target specified, use flat
        target = FrequencyResponse(name='room-target')
        target.raw = np.zeros(target.frequency.shape)
        target.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)
    return target


def open_mic_calibration(estimator, dir_path, mic_calibration=None):
    """Opens room measurement microphone calibration file

    Args:
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to directory
        mic_calibration: Path to explicitly given (if any) room measurement microphone calibration file

    Returns:
        Microphone calibration FrequencyResponse
    """
    if mic_calibration is None:
        # Room mic calibration file path not given, try csv first then txt
        mic_calibration = os.path.join(dir_path, 'room-mic-calibration.csv')
        if not os.path.isfile(mic_calibration):
            mic_calibration = os.path.join(dir_path, 'room-mic-calibration.txt')
    elif not os.path.isfile(mic_calibration):
        # Room mic calibration file path given, but the file doesn't exist
        raise FileNotFoundError(f'Room mic calibration file doesn\'t exist at "{mic_calibration}"')
    if os.path.isfile(mic_calibration):
        # File found, create frequency response
        mic_calibration = FrequencyResponse.read_from_csv(mic_calibration)
        mic_calibration.interpolate(f_step=1.01, f_min=10, f_max=estimator.fs / 2)
        mic_calibration.center()
    else:
        # File not found, skip calibration
        mic_calibration = None
    return mic_calibration
