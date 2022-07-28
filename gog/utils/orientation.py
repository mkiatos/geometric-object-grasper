import numpy as np
import numpy.linalg as lin
import math
from typing import Union, Tuple, List


class Affine3:
    def __init__(self):
        self.linear = np.eye(3)
        self.translation = np.zeros(3)

    @classmethod
    def from_matrix(cls, matrix):
        assert isinstance(matrix, np.ndarray)
        result = cls()
        result.translation = matrix[:3, 3]
        result.linear = matrix[0:3, 0:3]
        return result

    @classmethod
    def from_vec_quat(cls, pos, quat):
        assert isinstance(pos, np.ndarray)
        assert isinstance(quat, Quaternion)
        result = cls()
        result.translation = pos.copy()
        result.linear = quat.rotation_matrix()
        return result

    def matrix(self):
        matrix = np.eye(4)
        matrix[0:3, 0:3] = self.linear
        matrix[0:3, 3] = self.translation
        return matrix

    def quat(self):
        return Quaternion.from_rotation_matrix(self.linear)

    def __copy__(self):
        result = Affine3()
        result.linear = self.linear.copy()
        result.translation = self.translation.copy()
        return result

    def copy(self):
        return self.__copy__()

    def inv(self):
        return Affine3.from_matrix(np.linalg.inv(self.matrix()))

    def __mul__(self, other):
        return self.__rmul__(other)

    def __rmul__(self, other):
        return Affine3.from_matrix(np.matmul(self.matrix(), other.matrix()))

    def __str__(self):
        return self.matrix().__str__()


class Quaternion:

    # ============== Constructors ================

    def __init__(self, w=1., x=0., y=0., z=0.):
        """
        Constructs a quaternion. The quaternion is not normalized.

        Arguments:
        w -- float, the scalar part
        x -- float, the element along the x-axis of the vector part
        y -- float, the element along the y-axis of the vector part
        z -- float, the element along the z-axis of the vector part
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_vector(cls, vector, convention='wxyz'):
        """
        Constructs a quaternion from a vector.

        Arguments:
        vector -- vector with 4 elements (must implement operator '[i]')
        convention -- 'wxyz' or 'xyzw' corresponding to the order of the elements in 'vector'

        Returns:
        A Quaterion object.
        """
        if convention == 'wxyz':
            return cls(w=vector[0], x=vector[1], y=vector[2], z=vector[3])
        elif convention == 'xyzw':
            return cls(w=vector[3], x=vector[0], y=vector[1], z=vector[2])
        else:
            raise ValueError('Order is not supported.')

    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Constructs a quaternion from a rotation matrix.

        Arguments:
        R -- rotation matrix as 3x3 array-like object (must implement operator '[i,j]')

        Returns:
        A Quaterion object.
        """

        q = [None] * 4

        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S=4*qwh
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S

        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S

        result = q / np.linalg.norm(q)
        return cls(w=result[0], x=result[1], y=result[2], z=result[3])

    @classmethod
    def from_roll_pitch_yaw(cls, x):
        """
        Constructs a quaternion from roll-pitch-yaw angles.

        Arguments:
        x -- stores the roll-pitch-yaw angles (must implement operator '[i]')

        Returns:
        A Quaterion object.
        """
        roll = x[0]
        pitch = x[1]
        yaw = x[2]

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        return cls(w=cr * cp * cy + sr * sp * sy,
                   x=sr * cp * cy - cr * sp * sy,
                   y=cr * sp * cy + sr * cp * sy,
                   z=cr * cp * sy - sr * sp * cy)

    # ============== Misc ================

    def __getitem__(self, item):
        """
        Applies the [] operator to the quaternion, treating it as an array [w, x, y, z].

        Arguments:
        item -- int between [0, 3] or any other valid slice

        Returns:
        The element at position 'item' or the corresponding slice.
        """

        quat = np.array([self.w, self.x, self.y, self.z])
        return quat[item]

    def __copy__(self):
        """
        Returns a deep copy of 'self'.
        """
        return Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)

    def copy(self):
        return self.__copy__()

    def __call__(self, convention='wxyz') -> np.array:
        """
        See @self.as_vector
        """

        return self.as_vector(convention)

    def as_vector(self, convention='wxyz') -> np.array:
        """
        Returns the quaternion as an np.array.

        Arguments:
        convention -- 'wxyz' or xyzw'

        Returns:
        The quaternion as an np.array in the format defined by 'convention'.
        """
        if convention == 'wxyz':
            return np.array([self.w, self.x, self.y, self.z])
        elif convention == 'xyzw':
            return np.array([self.x, self.y, self.z, self.w])
        else:
            raise RuntimeError

    def vec(self) -> np.array:
        """
        Returns: np.array(3), the vector part of the quaternion.
        """
        return np.array([self.x, self.y, self.z])

    def __str__(self):
        return '%.3f' % self.w + " + " + '%.3f' % self.x + "i +" + '%.3f' % self.y + "j + " + '%.3f' % self.z + "k"

    # ============== Mathematical operations ================

    def mul(self, quat2: 'Quaternion'):
        """
        Implements the quaternion product between 'self' and another quaternion.

        Arguments:
        quat2 -- Quaternion

        Returns:
        Quaternion, the quaternion product of 'self' and 'quat2'
        """

        assert isinstance(quat2, Quaternion), "'second' must be a Quaternion"

        w1, v1 = self.w, self.vec()
        w2, v2 = quat2.w, quat2.vec()
        w = w1 * w2 - np.dot(v1, v2)
        v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
        return Quaternion(w, *v)

    def inv(self):
        """
        Returns: Quaternion, the quaternion inverse of 'self'.
        """

        temp = pow(self.w, 2) + pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2)
        return Quaternion(self.w/temp, -self.x/temp, -self.y/temp, -self.z/temp)

    def diff(self, quat2: 'Quaternion'):
        """
        Calculates the quaternion difference, i.e. the product of 'self' with inverse('quat2').

        Arguments:
        quat2 -- Quaternion

        Returns:
        Quaternion, the difference between 'self' and 'quat2'.

        Note: Since quat2 and -quat2 represent the same orientation, the minimum difference between 'self' and 'quat2'
        is calculated.
        """

        assert isinstance(quat2, Quaternion), "'quat2' must be a Quaternion"

        if np.dot(self(), quat2()) < 0:
            quat2 = quat2.copy().negate()

        return self.mul(quat2.inv())

    @staticmethod
    def log(quat: 'Quaternion', zero_tol=1e-16):
        """
        Calculates the quaternion logarithm as 2*log(quat)

        Arguments:
        quat -- A @Quaternion object.
        zero_tol -- Zero tolerance threshold (optional, default=1e-16)

        Returns:
        qlog -- np.array(3), the quaternion logarithm of quat.
        """

        assert isinstance(quat, Quaternion), "'quat' must be a Quaternion"

        v = quat.vec()
        v_norm = np.linalg.norm(v)

        if v_norm > zero_tol:
            qlog = 2 * math.atan2(v_norm, quat.w) * v / v_norm
        else:
            qlog = np.array([0, 0, 0])

        return qlog

    @staticmethod
    def exp(qlog: np.array, zero_tol=1e-16):
        """
        Calculates the quaternion exponential as exp(2*qlog)

        Arguments:
        qlog -- np.array(3)
        zero_tol -- Zero tolerance threshold (optional, default=1e-16)

        Returns:
        quat -- @Quaternion, the quaternion exponential of qlog.
        """

        norm_qlog = np.linalg.norm(qlog)
        theta = norm_qlog

        if theta > zero_tol:
            quat = Quaternion(math.cos(theta / 2.), *(math.sin(theta / 2) * qlog / norm_qlog))
        else:
            quat = Quaternion(1, 0, 0, 0)

        return quat

    @staticmethod
    def logDot_to_rotVel(logQ_dot: np.array, quat: 'Quaternion'):
        """
        Calculates the rotational velocity, corresponding to dlog(Q)/dt.

        Arguments:
            logQ_dot -- np.array(3), the 1st order time derivative of the quaternion logarithm of 'Q'
            Q -- Quaternion (must be unit quaternion!)

        Returns:
            rot_vel -- np.array(3), rotational velocity corresponding to dlog(Q)/dt
        """

        assert isinstance(quat, Quaternion), "'quat' must be a Quaternion"

        logQ_dot = logQ_dot.squeeze()  # just to make sure...
        assert logQ_dot.shape[0] == 3, 'logQ_dot must be an np.array(3)'

        # return np.matmul(jacob_rotVel_qLog(Q), logQ_dot)

        w = quat.w
        if (1 - math.fabs(w)) <= 1e-8:
            rot_vel = logQ_dot.copy()
        else:
            v = quat.vec()
            norm_v = np.linalg.norm(v)
            k = v / norm_v  # axis of rotation
            s_th = norm_v  # sin(theta/2)
            c_th = w  # cos(theta/2)
            th = math.atan2(s_th, c_th)  # theta/2
            Pk_qdot = k * np.dot(k, logQ_dot)  # projection of logQ_dot on k
            rot_vel = Pk_qdot + (logQ_dot - Pk_qdot) * s_th * c_th / th + (s_th ** 2 / th) * np.cross(k, logQ_dot)

        return rot_vel

    @staticmethod
    def logDDot_to_rotAccel(logQ_ddot: np.array, rotVel: np.array, quat: 'Quaternion'):

        assert isinstance(quat, Quaternion), "'quat' must be a Quaternion"

        logQ_ddot = logQ_ddot.squeeze()
        rotVel = rotVel.squeeze()

        w = quat.w
        if (1-math.fabs(w)) <= 1e-8:
            rot_accel = logQ_ddot
        else:
            v = quat.vec()
            norm_v = np.linalg.norm(v)
            k = v / norm_v # axis of rotation
            s_th = norm_v # sin(theta/2)
            c_th = w     # cos(theta/2)
            th = math.atan2(s_th, c_th) # theta/2

            Pk_rotVel = k*np.dot(k, rotVel) # projection of rotVel on k
            qdot = Pk_rotVel + (rotVel - Pk_rotVel)*th*c_th/s_th - th*np.cross(k, rotVel)

            qdot_bot = qdot - np.dot(k, qdot)*k # projection of qdot on plane normal to k
            k_dot = 0.5*qdot_bot/th
            th2_dot = 0.5*np.dot(k, qdot)

            sc_over_th = (s_th * c_th) / th

            JnDot_qdot = (1 - sc_over_th)*(np.dot(k, qdot)*k_dot + np.dot(k_dot, qdot)*k) + \
                    (s_th**2/th)*np.cross(k_dot, qdot) + \
                    ( (1 - 2*s_th**2)/th - sc_over_th/th )*th2_dot*qdot_bot + \
                    (2*sc_over_th - (s_th/th)**2)*th2_dot*np.cross(k, qdot)

            Pk_qddot = np.dot(k, logQ_ddot)*k # projection of qddot on k
            Jn_qddot = Pk_qddot + (logQ_ddot - Pk_qddot)*sc_over_th + (s_th**2/th)*np.cross(k, logQ_ddot)

            rot_accel = Jn_qddot + JnDot_qdot

        return rot_accel

    def normalize(self) -> 'Quaternion':
        """
        Normalizes 'self' so that it is a unit quaternion.
        """
        q = self.as_vector()
        q = q / np.linalg.norm(q)
        self.w = q[0]
        self.x = q[1]
        self.y = q[2]
        self.z = q[3]

        return self

    def negate(self) -> 'Quaternion':
        """
        Inverts the sign of all elements of 'self'. Notice that the represented orientation doesn't alter.
        """

        self.w, self.x, self.y, self.z = -self.w, -self.x, -self.y, -self.z
        return self

    # ============== Conversions to other orientation representations ================

    def rotation_matrix(self) -> np.array:
        """
        Returns: np.array(3), the rotation matrix of this quaternion.
        """
        n = self.w
        ex = self.x
        ey = self.y
        ez = self.z

        R = np.eye(3)

        R[0, 0] = 2 * (n * n + ex * ex) - 1
        R[0, 1] = 2 * (ex * ey - n * ez)
        R[0, 2] = 2 * (ex * ez + n * ey)

        R[1, 0] = 2 * (ex * ey + n * ez)
        R[1, 1] = 2 * (n * n + ey * ey) - 1
        R[1, 2] = 2 * (ey * ez - n * ex)

        R[2, 0] = 2 * (ex * ez - n * ey)
        R[2, 1] = 2 * (ey * ez + n * ex)
        R[2, 2] = 2 * (n * n + ez * ez) - 1

        return R

    def roll_pitch_yaw(self) -> Tuple[float, float, float]:
        # Calculate q2 using lots of information in the rotation matrix.
        # Rsum = abs( cos(q2) ) is inherently non-negative.
        # R20 = -sin(q2) may be negative, zero, or positive.
        R = self.rotation_matrix()
        R22 = R[2, 2]
        R21 = R[2, 1]
        R10 = R[1, 0]
        R00 = R[0, 0]
        Rsum = np.sqrt((R22 * R22 + R21 * R21 + R10 * R10 + R00 * R00) / 2)
        R20 = R[2, 0]
        q2 = np.arctan2(-R20, Rsum)

        e0 = self.w
        e1 = self.x
        e2 = self.y
        e3 = self.z
        yA = e1 + e3
        xA = e0 - e2
        yB = e3 - e1
        xB = e0 + e2
        epsilon = 1e-10
        isSingularA = (np.abs(yA) <= epsilon) and (np.abs(xA) <= epsilon)
        isSingularB = (np.abs(yB) <= epsilon) and (np.abs(xB) <= epsilon)
        if isSingularA:
            zA = 0.0
        else:
            zA = np.arctan2(yA, xA)
        if isSingularB:
            zB = 0.0
        else:
            zB = np.arctan2(yB, xB)
        q1 = zA - zB
        q3 = zA + zB

        # If necessary, modify angles q1 and/or q3 to be between -pi and pi.
        if q1 > np.pi:
            q1 = q1 - 2 * np.pi
        if q1 < -np.pi:
            q1 = q1 + 2 * np.pi
        if q3 > np.pi:
            q3 = q3 - 2 * np.pi
        if q3 < -np.pi:
            q3 = q3 + 2 * np.pi

        return q1, q2, q3


def rot_x(theta):
    rot = np.zeros((3, 3))
    rot[0, 0] = 1
    rot[0, 1] = 0
    rot[0, 2] = 0

    rot[1, 0] = 0
    rot[1, 1] = math.cos(theta)
    rot[1, 2] = - math.sin(theta)

    rot[2, 0] = 0
    rot[2, 1] = math.sin(theta)
    rot[2, 2] = math.cos(theta)

    return rot


def rot_y(theta):
    rot = np.zeros((3, 3))
    rot[0, 0] = math.cos(theta)
    rot[0, 1] = 0
    rot[0, 2] = math.sin(theta)

    rot[1, 0] = 0
    rot[1, 1] = 1
    rot[1, 2] = 0

    rot[2, 0] = - math.sin(theta)
    rot[2, 1] = 0
    rot[2, 2] = math.cos(theta)

    return rot


def rot_z(theta):
    rot = np.zeros((3, 3))
    rot[0, 0] = math.cos(theta)
    rot[0, 1] = -math.sin(theta)
    rot[0, 2] = 0

    rot[1, 0] = math.sin(theta)
    rot[1, 1] = math.cos(theta)
    rot[1, 2] = 0

    rot[2, 0] = 0
    rot[2, 1] = 0
    rot[2, 2] = 1

    return rot


def skew_symmetric(vector):
    output = np.zeros((3, 3))
    output[0, 1] = -vector[2]
    output[0, 2] = vector[1]
    output[1, 0] = vector[2]
    output[1, 2] = -vector[0]
    output[2, 0] = -vector[1]
    output[2, 1] = vector[0]
    return output


def angle_axis2rot(angle, axis):
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1 - c
    kx = axis[0]
    ky = axis[1]
    kz = axis[2]

    rot = np.eye(3)
    rot[0, 0] = pow(kx, 2) * v + c
    rot[0, 1] = kx * ky * v - kz * s
    rot[0, 2] = kx * kz * v + ky * s

    rot[1, 0] = kx * ky * v + kz * s
    rot[1, 1] = pow(ky, 2) * v + c
    rot[1, 2] = ky * kz * v - kx * s

    rot[2, 0] = kx * kz * v - ky * s
    rot[2, 1] = ky * kz * v + kx * s
    rot[2, 2] = pow(kz, 2) * v + c

    return rot


def quat_product(quat_1, quat_2):
    # Computer the product of the two quaternions, term by term
    final_quat = Quaternion()
    final_quat.w = quat_1.w * quat_2.w - quat_1.x * quat_2.x - quat_1.y * quat_2.y - quat_1.z * quat_2.z
    final_quat.x = quat_1.w * quat_2.x + quat_1.x * quat_2.w + quat_1.y * quat_2.z - quat_1.z * quat_2.y
    final_quat.y = quat_1.w * quat_2.y - quat_1.x * quat_2.z + quat_1.y * quat_2.w + quat_1.z * quat_2.x
    final_quat.z = quat_1.w * quat_2.z + quat_1.x * quat_2.y - quat_1.y * quat_2.x + quat_1.z * quat_2.w

    return final_quat


def quat_exp(vec):
    quat = Quaternion()
    vec_norm = np.linalg.norm(vec)

    quat_vec = np.zeros(3)
    if vec_norm > 1e-8:
        quat_vec = np.sin(vec_norm) * vec / vec_norm

    quat.w = np.cos(vec_norm)
    quat.x = quat_vec[0]
    quat.y = quat_vec[1]
    quat.z = quat_vec[2]
    return quat


def log_error_to_angular_velocity_jacobian(quaternion_error, log_error):
    quat_epsilon = quaternion_error.vec()
    eta = quaternion_error.w

    if 1 - eta * eta > 1e-6:
        return (- 0.5 * eta * np.matmul(skew_symmetric(log_error), skew_symmetric(quat_epsilon)) +
                np.matmul(quat_epsilon, np.transpose(quat_epsilon))) / (1 - eta * eta) - 0.5 * skew_symmetric(log_error)
    else:
        return np.eye(3)