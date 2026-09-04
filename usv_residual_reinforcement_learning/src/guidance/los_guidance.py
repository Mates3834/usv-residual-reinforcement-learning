import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def los_guidance(position, path, waypoint_index, switch_radius=3.0):
    """
    Line-of-Sight waypoint guidance.

    Returns:
        psi_ref, updated_waypoint_index
    """
    path = np.asarray(path)
    position = np.asarray(position)

    waypoint_index = int(np.clip(waypoint_index, 0, len(path) - 1))
    wp = path[waypoint_index]

    if (
        np.linalg.norm(wp - position) < switch_radius
        and waypoint_index < len(path) - 1
    ):
        waypoint_index += 1
        wp = path[waypoint_index]

    delta = wp - position
    psi_ref = np.arctan2(delta[1], delta[0])

    return float(wrap_angle(psi_ref)), waypoint_index
