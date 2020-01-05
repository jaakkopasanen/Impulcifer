# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from autoeq.frequency_response import FrequencyResponse
from impulse_response import ImpulseResponse
from hrir import HRIR
from utils import sync_axes, save_fig_as_png, read_wav
from constants import SPEAKER_NAMES, SPEAKER_LIST_PATTERN, IR_ROOM_SPL


def room_correction(
        estimator,
        dir_path,
        target=None,
        mic_calibration=None,
        fr_combination_method='average',
        plot=False):
    """Corrects room acoustics

    Args:
        estimator: ImpulseResponseEstimator
        dir_path: Path to directory
        target: Path to room target response CSV file
        mic_calibration: Path to room measurement microphone calibration file
        fr_combination_method: Method for combining generic room measurment frequency responses. "average" or
                               "conservative"
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
    room_fr = open_generic_room_measurement(estimator, dir_path, mic_calibration, target, method=fr_combination_method)

    if not len(rir.irs) and room_fr is None:
        # No room recording files found
        return None, None

    # Crop heads and tails from room impulse responses
    for speaker, pair in rir.irs.items():
        for side, ir in pair.items():
            ir.crop_head()
    rir.crop_tails()
    rir.write_wav(os.path.join(dir_path, 'room-responses.wav'))

    figs = None
    if plot:
        # Plot all but frequency response
        plot_dir = os.path.join(dir_path, 'plots', 'room')
        os.makedirs(plot_dir, exist_ok=True)
        figs = rir.plot(plot_fr=False, close_plots=False)

    # Create equalization frequency responses
    reference_gain = None
    fr_axes = []
    frs = dict()
    for speaker, pair in rir.irs.items():
        frs[speaker] = dict()
        for side, ir in pair.items():
            # Create frequency response
            fr = ir.frequency_response()

            if mic_calibration is not None:
                # Calibrate frequency response
                fr.raw -= mic_calibration.raw

            # Process
            fr.compensate(target, min_mean_error=True)

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

            # Add frequency response
            frs[speaker][side] = fr

            if plot:
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                fr = fr.copy()
                fr.smoothen_heavy_light()
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

    # Use generic measurement for speakers that don't have specific measurements
    for speaker in missing:
        frs[speaker] = {'left': room_fr.copy(), 'right': room_fr.copy()}

    if plot:
        # Sync FR plot axes
        sync_axes(fr_axes)
        # Save figures
        for speaker, pair in figs.items():
            for side, fig in pair.items():
                file_path = os.path.join(dir_path, 'plots', 'room', f'{speaker}-{side}.png')
                os.makedirs(os.path.split(file_path)[0], exist_ok=True)
                save_fig_as_png(file_path, fig)
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
    pattern = r'^room-{pattern}(-(left|right))?\.wav$'.format(pattern=SPEAKER_LIST_PATTERN)
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


def open_generic_room_measurement(estimator, dir_path, mic_calibration, target, method='average'):
    """Opens generic room measurment file

    Args:
        estimator: ImpulseResponseEstimator instance
        dir_path: Path to directory
        mic_calibration: Measurement microphone calibration FrequencyResponse
        target: Room response target FrequencyResponse
        method: Combination method. "average" or "conservative"

    Returns:

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
        ir = ImpulseResponse(estimator.estimate(track), estimator.fs, track)
        # Crop harmonic distortion from the head
        # Noise in the tail should not affect frequency response so it doesn't have to be cropped
        ir.crop_head(head_ms=1)
        irs.append(ir)

    # Frequency response for the generic room measurement
    generic_room_fr = FrequencyResponse(
        name='generic_room',
        frequency=FrequencyResponse.generate_frequencies(f_min=10, f_max=estimator.fs / 2, f_step=1.01),
        error=0
    )

    # Calculate and stack errors
    errors = []
    for ir in irs:
        fr = ir.frequency_response()
        if mic_calibration is not None:
            fr.raw -= mic_calibration.raw
        fr.compensate(target, min_mean_error=True)
        errors.append(fr.error_smoothed)
    errors = np.vstack(errors)

    # Combine errors
    if method == 'conservative':
        # Conservative error curve is zero everywhere else but on indexes where both have the same sign,
        # at these indexes the smaller absolute value is selected.
        # This ensures that no curve will be adjusted to the other side of zero
        mask = np.mean(errors > 0, axis=0)  # Average from boolean values per column
        positive = mask == 1  # Mask for columns with only positive values
        negative = mask == 0  # Mask for columns with only negative values
        # Minimum value for columns with only positive values
        generic_room_fr.error[positive] = np.min(errors[:, positive], axis=0)
        # Maximum value for columns with only negative values (minimum absolute value)
        generic_room_fr.error[negative] = np.max(errors[:, negative], axis=0)
        # Smoothen out kinks
        generic_room_fr.smoothen_fractional_octave(window_size=1 / 3, treble_window_size=1 / 3)
        generic_room_fr.error = generic_room_fr.error_smoothed.copy()
        generic_room_fr.error_smoothed = np.array([])

    elif method == 'average':
        generic_room_fr.error = np.mean(errors, axis=0)

    else:
        raise ValueError(
            f'Invalid value "{method}" for method. Supported values are "conservative" and "average"')

    # Zero error above 1 kHz
    start = np.argmax(generic_room_fr.frequency > 500)
    end = np.argmax(generic_room_fr.frequency > 1000)
    mask = np.concatenate([
        np.zeros(start - 1 if start > 0 else 0),
        signal.windows.hann(end - start),
        np.zeros(len(generic_room_fr.frequency) - end)
    ])
    generic_room_fr.error *= mask

    return generic_room_fr


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
