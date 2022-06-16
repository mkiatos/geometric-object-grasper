import numpy as np
import numpy.linalg as lin
import math


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
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __copy__(self):
        return Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)

    def copy(self):
        return self.__copy__()

    def as_vector(self, convention='wxyz'):
        if convention == 'wxyz':
            return np.array([self.w, self.x, self.y, self.z])
        elif convention == 'xyzw':
            return np.array([self.x, self.y, self.z, self.w])
        else:
            raise RuntimeError

    def normalize(self):
        q = self.as_vector()
        q = q / np.linalg.norm(q)
        self.w = q[0]
        self.x = q[1]
        self.y = q[2]
        self.z = q[3]

    def vec(self):
        return np.array([self.x, self.y, self.z])

    def rotation_matrix(self):
        """
        Transforms a quaternion to a rotation matrix.
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

    def euler_angles(self):
        """
        Transforms a quaternion to euler angles
        """
        n = self.w
        ex = self.x
        ey = self.y
        ez = self.z

        t0 = +2.0 * (n * ex + ey * ez)
        t1 = +1.0 - 2.0 * (ex * ex + ey * ey)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (n * ey - ez * ex)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (n * ez + ex * ey)
        t4 = +1.0 - 2.0 * (ey * ey + ez * ez)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    @classmethod
    def from_euler_angles(cls, roll, pitch, yaw):
        """
        Transforms euler angles to a quaternion
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        return cls(x=qx, y=qy, z=qz, w=qw)

    @classmethod
    def from_vector(cls, vector, order='wxyz'):
        if order == 'wxyz':
            return cls(w=vector[0], x=vector[1], y=vector[2], z=vector[3])
        elif order == 'xyzw':
            return cls(w=vector[3], x=vector[0], y=vector[1], z=vector[2])
        else:
            raise ValueError('Order is not supported.')

    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Transforms a rotation matrix to a quaternion.
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

        result = q / lin.norm(q);
        return cls(w=result[0], x=result[1], y=result[2], z=result[3])

    def __str__(self):
        return str(self.w) + " + " + str(self.x) + "i +" + str(self.y) + "j + " + str(self.z) + "k"

    def rot_z(self, theta):
        mat = self.rotation_matrix()
        mat =  np.matmul(mat, rot_z(theta))
        new = Quaternion.from_rotation_matrix(mat)
        self.w = new.w
        self.x = new.x
        self.y = new.y
        self.z = new.z
        return self

    def log(self):
        if abs(self.w - 1) < 1e-12:
            return np.zero(3)
        vec_norm = lin.norm(self.vec())
        return math.atan2(vec_norm, self.w) * self.vec() / vec_norm

    def mul(self, second):
        result = Quaternion()
        result.w = self.w * second.w - np.dot(self.vec(), second.vec())
        vec = self.w * second.vec() + second.w * self.vec() + np.cross(self.vec(), second.vec())
        result.x = vec[0]
        result.y = vec[1]
        result.z = vec[2]
        return result

    def inverse(self):
        result = Quaternion();
        temp = pow(self.w, 2) + pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2)
        result.w = self.w / temp
        result.x = - self.x / temp
        result.y = - self.y / temp
        result.z = - self.z / temp
        return result

    def log_error(self, desired):
        diff = self.mul(desired.inverse())
        if diff.w < 0:
            diff.x = - diff.x
            diff.y = - diff.y
            diff.z = - diff.z
            diff.w = - diff.w
        return 2.0 * diff.log()


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