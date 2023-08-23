import numpy as np


class EyeGazeFeatures:
    def __init__(self, gaze_angle_x, gaze_angle_y, timestamps, gaze_degree_error=9):
        self._gaze_angle_x = gaze_angle_x
        self._gaze_angle_y = gaze_angle_y
        self._timestamps = timestamps
        self._gaze_degree_error = gaze_degree_error

        self._features = {}

    def run(self):
        self._add_to_features('angle_x', self._gaze_angle_x)

        fixation_durations = compute_fixation_durations(self._gaze_angle_x, self._gaze_angle_y,
                                                        self._timestamps, self._gaze_degree_error)
        self._add_to_features('fixation_duration', fixation_durations)

        saccade_amplitudes = compute_saccade_amplitude(self._gaze_angle_x, self._gaze_angle_y)
        self._add_to_features('saccade_amplitude', saccade_amplitudes)

        velocities, accelerations = compute_velocity_acceleration(self._gaze_angle_x, self._gaze_angle_x,
                                                                  self._timestamps)
        self._add_to_features('velocity', velocities)
        self._add_to_features('acceleration', accelerations)

        return self._features

    def _add_to_features(self, name, values):
        mean = np.mean(values)
        std = np.std(values)
        self._features[f'gaze_mean_{name}'] = 0 if np.isnan(mean) else mean
        self._features[f'gaze_std_{name}'] = 0 if np.isnan(std) else std


def compute_fixation_durations(gaze_angle_x, gaze_angle_y, timestamps, threshold):
    """
    Compute fixation durations based on gaze angle data.
    :param gaze_angle_x: List of gaze_angle_x values
    :param gaze_angle_y: List of gaze_angle_y values
    :param timestamps: List of timestamps corresponding to gaze_angle_x and gaze_angle_y
    :param threshold: Angle threshold to consider as a fixation
    :return: List of fixation durations
    """
    fixation_durations = []
    current_fixation_duration = 0

    for i in range(1, len(gaze_angle_x)):
        dx = abs(gaze_angle_x[i] - gaze_angle_x[i - 1])
        dy = abs(gaze_angle_y[i] - gaze_angle_y[i - 1])

        if dx < np.radians(threshold) and dy < np.radians(threshold):
            current_fixation_duration += timestamps[i] - timestamps[i - 1]
        else:
            if current_fixation_duration > 0:
                fixation_durations.append(current_fixation_duration)
                current_fixation_duration = 0

    # Add the last fixation duration if it exists
    if current_fixation_duration > 0:
        fixation_durations.append(current_fixation_duration)

    return fixation_durations


def compute_saccade_amplitude(gaze_angle_x, gaze_angle_y):
    """
    Compute saccade amplitude based on gaze angle data.
    :param gaze_angle_x: List of gaze_angle_x values
    :param gaze_angle_y: List of gaze_angle_y values
    :return: List of saccade amplitudes
    """
    gaze_angles = np.array([gaze_angle_x, gaze_angle_y])
    saccade_amplitudes = np.linalg.norm(np.diff(gaze_angles, axis=1), axis=0)

    return saccade_amplitudes


def compute_velocity_acceleration(gaze_angle_x, gaze_angle_y, timestamps):
    gaze_angles = np.array([gaze_angle_x, gaze_angle_y])
    # Calculate the velocity and acceleration of gaze movements
    velocity = np.linalg.norm(np.diff(gaze_angles, axis=1) / np.diff(timestamps), axis=0)
    acceleration = np.diff(velocity, axis=0) / np.diff(timestamps[1:])

    return velocity, acceleration
