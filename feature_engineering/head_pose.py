import numpy as np


class HeadPoseFeatures:
    def __init__(self, pose_Rx, pose_Ry, pose_Rz, timestamps, head_degree_error=3):
        self._pose_Rx = pose_Rx
        self._pose_Ry = pose_Ry
        self._pose_Rz = pose_Rz
        self._timestamps = timestamps
        self._head_degree_error = head_degree_error

        self._features = {}

    def run(self):
        self._add_to_features('roll', self._pose_Rz)
        self._add_to_features('yaw', self._pose_Ry)

        movement, stability = compute_head_movement_stability_durations(self._pose_Rx, self._pose_Ry, self._pose_Rz,
                                                                        self._timestamps, self._head_degree_error)
        self._add_to_features('movement_duration', movement)
        self._add_to_features('stability_duration', stability)

        velocities, accelerations = compute_velocity_acceleration(self._pose_Rx, self._pose_Ry, self._pose_Rz,
                                                                  self._timestamps)
        self._add_to_features('velocity', velocities)
        self._add_to_features('acceleration', accelerations)

        return self._features

    def _add_to_features(self, name, values):
        mean = np.mean(values)
        std = np.std(values)
        self._features[f'head_mean_{name}'] = 0 if np.isnan(mean) else mean
        self._features[f'head_std_{name}'] = 0 if np.isnan(std) else std


def compute_head_movement_stability_durations(pose_Rx, pose_Ry, pose_Rz, timestamps, threshold):
    """
    Compute head movement and head stability durations based on head pose data.
    :param pose_Rx: List of pose_Rx values (pitch, in radians)
    :param pose_Ry: List of pose_Ry values (yaw, in radians)
    :param pose_Rz: List of pose_Rz values (roll, in radians)
    :param timestamps: List of timestamps corresponding to pose_Rx, pose_Ry, and pose_Rz
    :param threshold: Angle threshold (in degrees) to consider as movement (default: 10)
    :return: Two lists: head_movement_durations and head_stability_durations
    """
    head_movement_durations = []
    head_stability_durations = []
    current_movement_duration = 0
    current_stability_duration = 0

    for i in range(1, len(pose_Rx)):
        dx = abs(pose_Rx[i] - pose_Rx[i - 1])
        dy = abs(pose_Ry[i] - pose_Ry[i - 1])
        dz = abs(pose_Rz[i] - pose_Rz[i - 1])

        if (dx >= np.radians(threshold)) or (dy >= np.radians(threshold)) or (dz >= np.radians(threshold)):
            current_movement_duration += timestamps[i] - timestamps[i - 1]

            # Save the current stability duration if it exists
            if current_stability_duration > 0:
                head_stability_durations.append(current_stability_duration)
                current_stability_duration = 0
        else:
            current_stability_duration += timestamps[i] - timestamps[i - 1]

            # Save the current movement duration if it exists
            if current_movement_duration > 0:
                head_movement_durations.append(current_movement_duration)
                current_movement_duration = 0

    # Add the last movement or stability duration if it exists
    if current_movement_duration > 0:
        head_movement_durations.append(current_movement_duration)
    elif current_stability_duration > 0:
        head_stability_durations.append(current_stability_duration)

    return head_movement_durations, head_stability_durations


def compute_velocity_acceleration(pose_Rx, pose_Ry, pose_Rz, timestamps):
    head_pose = np.array([pose_Rx, pose_Ry, pose_Rz])
    # Calculate the velocity and acceleration of head movements
    velocity = np.linalg.norm(np.diff(head_pose, axis=1) / np.diff(timestamps), axis=0)
    acceleration = np.diff(velocity, axis=0) / np.diff(timestamps[1:])

    return velocity, acceleration
