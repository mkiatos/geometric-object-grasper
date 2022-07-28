import numpy as np
from typing import Union, Tuple, List
from gog.utils.orientation import Quaternion


def fifth_order_traj(t: float,
                     p0: Union[float, np.array],
                     pf: Union[float, np.array],
                     total_time: float) -> (Union[float, np.array], Union[float, np.array], Union[float, np.array]):

    if np.isscalar(p0):
        pos = p0
        vel = 0
        accel = 0
    else:
        n_dof = len(p0)
        pos = p0
        vel = np.zeros(n_dof)
        accel = np.zeros(n_dof)

    if t < 0:
        pos = p0
    elif t > total_time:
        pos = pf
    else:
        pos = p0 + (pf - p0) * (10 * pow(t / total_time, 3) -
                                15 * pow(t / total_time, 4) + 6 * pow(t / total_time, 5))
        vel = (pf - p0) * (30 * pow(t, 2) / pow(total_time, 3) -
                           60 * pow(t, 3) / pow(total_time, 4) + 30 * pow(t, 4) / pow(total_time, 5))
        accel = (pf - p0) * (60 * t / pow(total_time, 3) -
                             180 * pow(t, 2) / pow(total_time, 4) + 120 * pow(t, 3) / pow(total_time, 5))

    return pos, vel, accel


def fifth_order_quat_traj(t: float,
                          Q0: Quaternion,
                          Qf: Quaternion,
                          total_time: float) -> Tuple[Quaternion, np.array, np.array]:

    assert isinstance(Q0, Quaternion), "'Q0' must be a Quaternion"
    assert isinstance(Qf, Quaternion), "'Qf' must be a Quaternion"

    if np.dot(Q0.as_vector(), Qf.as_vector()) < 0:
        Qf.negate()

    qLog_0 = np.array([0, 0, 0])
    qLog_f = Quaternion.log(Qf.mul(Q0.inv()))

    logQ1, logQ1_dot, logQ1_ddot = fifth_order_traj(t, qLog_0, qLog_f, total_time)

    Q1 = Quaternion.exp(logQ1)
    Q = Q1.mul(Q0)
    rot_vel = Quaternion.logDot_to_rotVel(logQ1_dot, Q1)
    rot_accel = Quaternion.logDDot_to_rotAccel(logQ1_ddot, rot_vel, Q1)

    return Q, rot_vel, rot_accel
