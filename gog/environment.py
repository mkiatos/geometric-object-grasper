import matplotlib.pyplot as plt
import pybullet as p
import numpy as np
import os
import math
import time
import cv2
from typing import List

from gog.utils.orientation import Quaternion, Affine3,  rot_z,  rot_y
from gog.utils.trajectory import fifth_order_traj, fifth_order_quat_traj
from gog.utils import pybullet_utils as pybullet_utils
import gog.utils.utils as utils
from gog import cameras

MOUNT_URDF_PATH = 'mount.urdf'
UR5_URDF_PATH = 'ur5e_bhand.urdf'
UR5_WORKSPACE_URDF_PATH = 'table/table.urdf'
PLANE_URDF_PATH = "plane/plane.urdf"


class Object:
    def __init__(self,
                 name='',
                 pos=[0.0, 0.0, 0.0],
                 quat=Quaternion(),
                 size=[],
                 color=[1.0, 1.0, 1.0, 1.0],
                 body_id=None):
        """
          Represents an object.

          Parameters
          ----------
          name : str
              The name of the object
          pos : list
              The position of the object
          quat : Quaternion
              The orientation of the object in the form of quaternion
          body_id : int
              A unique id for the object
        """
        self.name = name
        self.pos = pos
        self.size = size
        self.home_quat = quat
        self.color = color
        self.body_id = body_id


class SimCamera:
    def __init__(self, config):
        self.pos = np.array(config['pos'])
        self.target_pos = np.array(config['target_pos'])
        self.up_vector = np.array(config['up_vector'])

        # Compute view matrix.
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=self.pos,
                                               cameraTargetPosition=self.target_pos,
                                               cameraUpVector=self.up_vector)

        self.z_near = config['zrange'][0]
        self.z_far = config['zrange'][1]
        self.width, self.height = config['image_size'][1], config['image_size'][0]
        self.fy = config['intrinsics'][0]

        # Compute projection matrix.
        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width / self.height,
                                                              nearVal=self.z_near, farVal=self.z_far)

    def get_pose(self):
        """
        Returns the camera pose w.r.t. world

        Returns
        -------
        np.array()
            4x4 matrix representing the camera pose w.r.t. world
        """
        return pybullet_utils.get_camera_pose(self.pos, self.target_pos, self.up_vector)

    def get_depth(self, depth_buffer):
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth

    def get_data(self):
        """
        Returns
        -------
        np.array(), np.array(), np.array()
            The rgb, depth and segmentation images
        """
        image = p.getCameraImage(self.width, self.height,
                                 self.view_matrix, self.projection_matrix,
                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return image[2], self.get_depth(image[3]), image[4]
# class Ur5Bhand:
#     def __init__(self, robot_id):
#         # TODO: hardcoded robot_id = 0
#         self.robot_id = robot_id
#         self.joint_names = {'robot': ['ur5_shoulder_pan_joint', 'ur5_shoulder_lift_joint', 'ur5_elbow_joint',
#                                       'ur5_wrist_1_joint', 'ur5_wrist_2_joint', 'ur5_wrist_3_joint'],
#                             'bhand': ['bh_j11_joint', 'bh_j21_joint', 'bh_j12_joint', 'bh_j22_joint',
#                                       'bh_j32_joint', 'bh_j13_joint', 'bh_j23_joint', 'bh_j33_joint']}
#
#         self.robot_indices = pybullet_utils.get_joint_indices(self.joint_names['robot'], robot_id)
#         self.bhand_indices = pybullet_utils.get_joint_indices(self.joint_names['bhand'], robot_id)
#         # self.ee_link_name = 'bh_palm_link'
#         self.ee_link_name = 'bh_base_link'
#         self.distal_links = ['bh_finger_13_link', 'bh_finger_23_link', 'bh_finger_33_link']
#
#         # Set robot and hand to position control
#         for joint in self.robot_indices:
#             p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=joint, controlMode=p.POSITION_CONTROL)
#
#         for joint in self.bhand_indices:
#             p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=joint, controlMode=p.POSITION_CONTROL)
#
#         # Set home configuration for robot and barrett
#         self.joint_configs = {"home": [-2.8927932236757625, -1.7518461461930528, -0.8471216131631573,
#                                        -2.1163833167682005, 1.5717067329577208, 0.2502483535771374],
#                               "above_table": [-2.8964885089272934, -1.7541597533564786, -1.9212388653019141,
#                                               -1.041716266062558, 1.5759665976832087, 0.24964880122853264]}
#         home_angle = 0.7 # 0.52
#         self.joint_configs = {"home": {'robot': self.joint_configs['home'],
#                                        'bhand': [0, 0., home_angle, home_angle, home_angle,
#                                                  home_angle/3, home_angle/3, home_angle/3]}}
#         self.move_home()
#
#     def get_joint_position(self):
#         robot_joint_pos = []
#         hand_joint_pos = []
#
#         for i in range(len(self.joint_names['robot'])):
#             robot_joint_pos.append(p.getJointState(0, self.robot_indices[i])[0])
#
#         for i in range(len(self.joint_names['bhand'])):
#             hand_joint_pos.append(p.getJointState(0, self.bhand_indices[i])[0])
#
#         return {'robot': robot_joint_pos, 'hand': hand_joint_pos}
#
#     def reset_robot_joint_position(self, joint_position):
#         for i in range(len(self.robot_indices)):
#             p.resetJointState(0, self.robot_indices[i], joint_position[i])
#         self.set_robot_joint_position(joint_position)
#
#     def set_robot_joint_position(self, joint_position):
#         for i in range(len(self.joint_names['robot'])):
#             p.setJointMotorControl2(bodyUniqueId=0,
#                                     jointIndex=self.robot_indices[i],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=joint_position[i])
#
#     def set_hand_joint_position(self, joint_position):
#         for i in range(len(self.joint_names['bhand'])):
#             p.setJointMotorControl2(bodyUniqueId=0,
#                                     jointIndex=self.bhand_indices[i],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=joint_position[i])
#
#     def get_task_pose(self):
#         return pybullet_utils.get_link_pose(self.ee_link_name)
#
#     def reset_task_pose(self, pos, quat):
#         link_index = pybullet_utils.get_link_indices([self.ee_link_name])[0]
#         joints = p.calculateInverseKinematics(bodyIndex=0,
#                                               endEffectorLinkIndex=link_index,
#                                               targetPosition=(pos[0], pos[1], pos[2]),
#                                               targetOrientation=quat.as_vector("xyzw"))
#         self.reset_robot_joint_position(joints)
#
#         # Inverse kinematics seems to not accurate when the target position is far from the current,
#         # resulting to errors after reset. Call trajectory to compensate for these errors
#         self.set_task_pose_trajectory(pos, quat, 0.2)
#
#     def set_task_pose(self, pos, quat):
#         link_index = pybullet_utils.get_link_indices([self.ee_link_name])[0]
#         joints = p.calculateInverseKinematics(bodyIndex=0,
#                                               endEffectorLinkIndex=link_index,
#                                               targetPosition=(pos[0], pos[1], pos[2]),
#                                               targetOrientation=quat.as_vector("xyzw"))
#         self.set_robot_joint_position(joints)
#
#     def set_task_pose_trajectory(self, pos, quat, duration):
#         def has_converged(quat, pos, t, duration):
#             robot_pos, robot_quat = self.get_task_pose()
#
#             quaternion_error = robot_quat.mul(quat.inverse())
#             if quaternion_error.w < 0:
#                 quaternion_error = - quaternion_error
#
#             orientation_error = 2 * quaternion_error.log()
#
#             if np.linalg.norm(orientation_error) < 0.1 and np.linalg.norm(pos - robot_pos) < 1e-2:
#                 return True
#             else:
#                 return False
#
#         init_pos, init_quat = self.get_task_pose()
#
#         trajectories = []
#         for i in range(3):
#             trajectories.append(robotics.Trajectory([0, duration], [init_pos[i], pos[i]]))
#
#         quat_error = init_quat.mul(quat.inverse())
#         if quat_error.w < 0:
#             quat_error = Quaternion(w=-quat_error.w, x=-quat_error.x, y=-quat_error.y, z=-quat_error.z)
#             quat = quat_error.inverse().mul(init_quat)
#
#         init_orientation_error = 2 * quat_error.log()
#
#         traj_orientation_error = []
#         for i in range(3):
#             traj_orientation_error.append(robotics.Trajectory([0, duration], [init_orientation_error[i], 0]))
#
#         t = 0
#         dt = 0.001
#
#         prev_quat = init_quat
#
#         while t < duration:
#             pos_d = np.zeros(3)
#
#             for i in range(3):
#                 pos_d[i] = trajectories[i].pos(t)
#
#             orientation_error = np.zeros(3)
#             for i in range(3):
#                 orientation_error[i] = traj_orientation_error[i].pos(t)
#
#             quaternion_error = quat_exp(0.5 * orientation_error)
#             quat_d = quaternion_error.mul(quat)
#             if np.dot(quat_d.vec(), prev_quat.vec()) + quat_d.w * prev_quat.w:
#                 quat_d = Quaternion(w=-quat_d.w, x=-quat_d.x, y=-quat_d.y, z=-quat_d.z)
#
#             prev_quat = quat_d
#
#             self.set_task_pose(pos_d, quat_d)
#
#             t += dt
#             p.stepSimulation()
#             time.sleep(dt)
#
#         if has_converged(quat, pos, t, duration):
#             print("I have converged!!!!!")
#         else:
#             print("No convergence")
#
#     def set_joints_clik(self, pos, quat, vel, angular_vel):
#         robot_pos, robot_quat = self.get_task_pose()
#
#         quaternion_error = robot_quat.mul(quat.inverse())
#         if quaternion_error.w < 0:
#             quaternion_error = - quaternion_error
#
#         orientation_error = 2 * quaternion_error.log()
#         position_error = robot_pos - pos
#
#         error = np.zeros(6)
#         error[0:3] = position_error
#         error[3:] = orientation_error
#
#         rotation_matrix_error = quaternion_error.rotation_matrix()
#
#         velocity = np.zeros(6)
#         velocity[0:3] = vel
#         velocity [3:] = np.matmul(rotation_matrix_error, angular_vel)
#
#         link_index = pybullet_utils.get_link_indices([self.ee_link_name])[0]
#         jacobian = p.calculateJacobian(bodyUniqueId=self.robot_id, linkIndex=link_index)
#
#         # joints_velocity =
#
#     def set_joint_trajectory(self, final, duration):
#         init = self.get_joint_position()['robot']
#         trajectories = []
#
#         for i in range(len(self.robot_indices)):
#             trajectories.append(robotics.Trajectory([0, duration], [init[i], final[i]]))
#
#         t = 0
#         dt = 0.001
#         while t < duration:
#             command = []
#             for i in range(len(self.robot_indices)):
#                 command.append(trajectories[i].pos(t))
#             self.set_robot_joint_position(command)
#             t += dt
#             p.stepSimulation()
#             time.sleep(dt)
#
#     def set_hand_joint_trajectory(self, final, duration):
#         init = self.get_joint_position()['hand']
#         final = [final[0], final[0], final[1], final[2], final[3], final[1]/3, final[2]/3, final[3]/3]
#
#         trajectories = []
#         for i in range(len(self.joint_names['bhand'])):
#             trajectories.append(robotics.Trajectory([0, duration], [init[i], final[i]]))
#
#         t = 0
#         dt = 0.001
#         while t < duration:
#             command = []
#             for i in range(len(self.joint_names['bhand'])):
#                 command.append(trajectories[i].pos(t))
#             self.set_hand_joint_position(command)
#
#             # ToDo: stop only one finger
#             for link in self.distal_links:
#                 link_index = pybullet_utils.get_link_indices([link])[0]
#                 points = p.getContactPoints(linkIndexA=link_index)
#                 stop_grasping = False
#                 for pnt in points:
#                     if pnt[9] > 15:
#                         stop_grasping = True
#                         break
#
#             if stop_grasping:
#                 break
#
#             t += dt
#             p.stepSimulation()
#             time.sleep(dt)
#
#     def step(self):
#         p.stepSimulation()
#
#     def teleport(self, pos, quat):
#         link_index = pybullet_utils.get_link_indices([self.ee_link_name])[0]
#         joint_position = p.calculateInverseKinematics(bodyIndex=0,
#                                                       endEffectorLinkIndex=link_index,
#                                                       targetPosition=(pos[0], pos[1], pos[2]),
#                                                       targetOrientation=quat.as_vector("xyzw"),
#                                                       lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
#                                                       upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
#                                                       jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
#                                                       )
#         self.reset_robot_joint_position(joint_position)
#
#     def move_home(self):
#         # Reset joints' state
#         for i in range(len(self.robot_indices)):
#             p.resetJointState(0, self.robot_indices[i], self.joint_configs["home"]['robot'][i])
#
#         for i in range(len(self.bhand_indices)):
#             p.resetJointState(0, self.bhand_indices[i], self.joint_configs["home"]['bhand'][i])
#
#         # Move to home
#         for i in range(len(self.robot_indices)):
#             p.setJointMotorControl2(bodyUniqueId=0,
#                                     jointIndex=self.robot_indices[i],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=self.joint_configs['home']['robot'][i])
#
#         for i in range(len(self.bhand_indices)):
#             p.setJointMotorControl2(bodyUniqueId=0, jointIndex=self.bhand_indices[i],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=self.joint_configs['home']['bhand'][i])


class FloatingGripper:

    def __init__(self, urdf_path: str, ee_link_name: str, finger_joints: List[str],
                 finger_default_joints: List[float] = [], base_pos=[0, 0, 0], base_quat=[0, 0, 0, 1]):

        self.body_id = None
        self.num_joints = None
        self.ee_link_ind = None
        self.joint_ind = None

        self.urdf = urdf_path

        # base_pos = [0, 0, 0]
        # base_quat_xyzw = [0, 0, 0, 1]
        self.body_id = p.loadURDF(self.urdf, basePosition=base_pos, baseOrientation=base_quat)

        joint_names = ['joint_x', 'joint_y', 'joint_z',
                       'joint_revolute_x', 'joint_revolute_y', 'joint_revolute_z']

        if not finger_default_joints:
            finger_default_joints = [0] * len(finger_joints)
        self.finger_joints_default_values = finger_default_joints.copy()
        joint_names = joint_names + finger_joints

        prefix = ''  # if a prefix is given...
        joint_names = [prefix + j_name for j_name in joint_names]
        ee_link_name = prefix + ee_link_name

        self.num_joints = len(joint_names)

        self.ee_link_ind = pybullet_utils.get_link_indices([ee_link_name], self.body_id)[0]
        self.joint_ind = pybullet_utils.get_joint_indices(joint_names, self.body_id)

        self.prev_rpy = [0, 0, 0]

        # Log the rpy values in case you want to plot them
        self.rpy_data = np.zeros((3, 1))

        p.setJointMotorControlArray(bodyIndex=self.body_id, jointIndices=self.joint_ind, controlMode=p.POSITION_CONTROL)

    def get_joint_position(self):
        return [p.getJointState(self.body_id, self.joint_ind[i])[0] for i in range(self.num_joints)]

    def get_joint_velocity(self):
        return [p.getJointState(self.body_id, self.joint_ind[i])[1] for i in range(self.num_joints)]

    def get_task_pose(self):
        state = p.getLinkState(self.body_id, self.ee_link_ind)
        pos = [state[0][0], state[0][1], state[0][2]]
        quat_wxyz = [state[1][3], state[1][0], state[1][1], state[1][2]]
        return pos, quat_wxyz

    def set_joint_position(self, joint_position):
        p.setJointMotorControlArray(bodyIndex=self.body_id, jointIndices=self.joint_ind,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_position)  # , forces=[2.0]*len(self.joint_ind))

    def reset_joint_position(self, joint_position):
        for i in range(len(self.joint_ind)):
            p.resetJointState(self.body_id, self.joint_ind[i], joint_position[i])

    def set_task_pose(self, pos, quat, finger_joints=None):
        if not finger_joints:
            finger_joints = self.finger_joints_default_values

        self.set_joint_position(self.calc_inv_kinematics(pos, quat) + finger_joints)

    def reset_task_pose(self, pos, quat, finger_joints=None):
        if not finger_joints:
            finger_joints = self.finger_joints_default_values

        pos_rpy = self.calc_inv_kinematics(pos, quat)
        self.reset_joint_position(pos_rpy + finger_joints)

    def calc_inv_kinematics(self, pos, quat):

        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # quat in xyzw format
        rpy = r.as_euler('zyx', degrees=False)
        rpy = [rpy[2], rpy[1], rpy[0]]

        for i in range(3):
            if abs(self.prev_rpy[i] - rpy[i]) > 6.0:
                # print('\33[1m\33[33mSingularity: prev=' + str(self.prev_rpy[i]) + ', current=' + str(rpy[i]))
                rpy[i] = 2 * math.pi + rpy[i]

        self.prev_rpy = rpy

        # Uncomment if you want to log the rpy values for plotting them later...
        # self.rpy_data = np.hstack([self.rpy_data, np.array(rpy).reshape(3, 1)])

        joints = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

        return joints


class FloatingBarret(FloatingGripper):

    def __init__(self, urdf_path: str, ee_link_name: str, prefix: str):
        finger_joints = ['_j11_joint', '_j21_joint', '_j12_joint', '_j22_joint',
                         '_j32_joint', '_j13_joint', '_j23_joint', '_j33_joint']
        finger_joints = [prefix + fing_name for fing_name in finger_joints]
        finger_default_joints = [0] * len(finger_joints)

        self.dt = 1.0 / 240

        base_pos = np.array([0.7, 0.0, 0.5])
        # base_quat_xyzw = Quaternion.from_rotation_matrix(rot_y(-np.pi)).as_vector('xyzw')

        self.joint_pos_data = np.zeros((6 + 4, 1))

        super().__init__(urdf_path, ee_link_name, finger_joints, finger_default_joints)

    def set_task_pose(self, pos, quat, finger_joints=None):
        """
        Arguments:
        pos -- np.array, the position of the gripper
        quat -- Quaternion, the orientation of the gripper
        finger_joints -- list[float], [pack, finger1, finger2, finger3] joints
        """

        if finger_joints is None:
            FloatingGripper.set_task_pose(self, pos, quat)
            return

        finger_joints_4 = finger_joints.tolist()

        mmult = 0.344262295082
        finger_joints = [finger_joints[0], 1.0 * finger_joints[0],
                         finger_joints[1], finger_joints[2], finger_joints[3],
                         mmult * finger_joints[1], mmult * finger_joints[2], mmult * finger_joints[3]]

        joints = np.array(self.calc_inv_kinematics(pos, quat) + finger_joints_4).reshape(-1, 1)
        self.joint_pos_data = np.hstack([self.joint_pos_data, joints])

        FloatingGripper.set_task_pose(self, pos, quat, finger_joints)

    def move(self, target_pos, target_quat, duration, plot_frames=False):

        p0, Q0 = self.get_task_pose()
        Q0 = Quaternion.from_vector(Q0, "wxyz")

        finger_joints = self.get_joint_position()[6:]
        finger_joints = np.array([finger_joints[0], finger_joints[2], finger_joints[3], finger_joints[4]])

        if plot_frames:
            count = 0

        t = 0
        while t < duration:

            p_ref = fifth_order_traj(t, p0, target_pos, duration)[0]
            Q_ref = fifth_order_quat_traj(t, Q0, target_quat, duration)[0]

            self.set_task_pose(p_ref, Q_ref.as_vector("wxyz"), finger_joints)

            if plot_frames:
                count += 1
                if count == 40:
                    # BulletFrame.remove("current")
                    pybullet_utils.BulletFrame.plot(p_ref, Q_ref, "ref", length=0.2, linewidth=3)
                    pybullet_utils.BulletFrame.plot(p0, Q0, "robot", length=0.4, linewidth=2)
                    count = 0

            t += self.dt
            p.stepSimulation()
            time.sleep(self.dt)

    def move_fingers(self, target_finger_joints, duration):
        finger_joints0 = self.get_joint_position()[6:]
        finger_joints0 = np.array([finger_joints0[0], finger_joints0[2], finger_joints0[3], finger_joints0[4]])

        p0, Q0 = self.get_task_pose()
        Q0 = Quaternion.from_vector(Q0, "wxyz")

        t = 0
        while t < duration:
            p_ref = fifth_order_traj(t, finger_joints0, target_finger_joints, duration)[0]

            self.set_task_pose(p0, Q0.as_vector("wxyz"), finger_joints=p_ref)  # [pack, finger1, finger2, finger3]

            t += self.dt
            p.stepSimulation()
            time.sleep(self.dt)

    def still(self):
        p0, Q0 = self.get_task_pose()
        Q0 = Quaternion.from_vector(Q0, "wxyz")

        finger_joints = self.get_joint_position()[6:]
        finger_joints = np.array([finger_joints[0], finger_joints[2], finger_joints[3], finger_joints[4]])

        self.set_task_pose(p0, Q0.as_vector("wxyz"), finger_joints)


class Environment:
    """
    Class that implements the environment in pyBullet.
    Parameters
    ----------
    """

    def __init__(self,
                 assets_root,
                 disp=True,
                 hz=240):

        self.pxl_size = 0.005
        self.bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]])  # workspace limits
        self.assets_root = assets_root

        self.nr_objects = [3, 4]
        self.disp = disp

        # Setup cameras.
        self.agent_cams = []
        for config in cameras.RealSense.CONFIG:
            config_world = config.copy()
            config_world['pos'] = config['pos']
            config_world['target_pos'] = config['target_pos']
            self.agent_cams.append(SimCamera(config_world))

        self.bhand = None
        self.objects = []
        self.obj_files = []
        objects_path = 'objects/unseen'
        for obj_file in os.listdir(os.path.join(assets_root, objects_path)):
            if not obj_file.endswith('.obj'):
                continue
            self.obj_files.append(os.path.join(assets_root, objects_path, obj_file))

        # Start PyBullet.
        if disp:
            p.connect(p.GUI)

            # Move default camera closer to the scene.
            target = np.array(np.zeros(3))
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=270,
                cameraPitch=-45,
                cameraTargetPosition=target)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(self.assets_root)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setTimeStep(1.0 / hz)
        self.dt = 1.0 / hz

        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng.seed(seed)

    @staticmethod
    def add_single_object(obj_path, pos, quat, size):
        body_id = pybullet_utils.load_obj(obj_path, scaling=1.0, position=pos, orientation=quat.as_vector("xyzw"))
        return Object(name=obj_path.split('/')[-1].split('.')[0],
                      pos=pos,
                      quat=quat,
                      size=size,
                      body_id=body_id)

    def add_objects(self):

        def get_pxl_distance(meters):
            return meters / self.pxl_size

        def get_xyz(pxl, obj_size):
            x = -(self.pxl_size * pxl[0] - self.bounds[0, 1])
            y = self.pxl_size * pxl[1] - self.bounds[1, 1]
            z = obj_size[2]
            return np.array([x, y, z])

        # Sample n objects from the database.
        nr_objects = self.rng.randint(low=self.nr_objects[0], high=self.nr_objects[1])
        obj_paths = self.rng.choice(self.obj_files, nr_objects)

        for i in range(len(obj_paths)):
            obj = Object()
            # base_position, base_orientation = self.workspace2world(np.array([1.0, 1.0, 0.0]), Quaternion())
            body_id = pybullet_utils.load_obj(obj_path=obj_paths[i], scaling=1.0, position=np.array([1.0, 1.0, 0.0]),
                                              orientation=Quaternion().as_vector("xyzw"))
            obj.body_id = body_id
            size = (np.array(p.getAABB(body_id)[1]) - np.array(p.getAABB(body_id)[0])) / 2.0

            max_size = np.sqrt(size[0] ** 2 + size[1] ** 2)
            erode_size = int(np.round(get_pxl_distance(meters=max_size)))

            obs = self.get_obs()
            state = utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)

            free = np.zeros(state.shape, dtype=np.uint8)
            free[state == 0] = 1
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0

            # Do not place object next to table limits
            free[0:20] = 0
            free[80:100] = 0
            free[:, 0:20] = 0
            free[:, 80:100] = 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))

            if np.sum(free) == 0:
                return
            pixx = utils.sample_distribution(np.float32(free), self.rng)
            pix = np.array([pixx[1], pixx[0]])

            pos = get_xyz(pix, size)
            theta = self.rng.uniform(0, 2 * np.pi)
            quat = Quaternion().from_rotation_matrix(rot_z(theta))

            p.removeBody(body_id)

            self.objects.append(self.add_single_object(obj_paths[i], pos, quat, size))

    def reset(self):
        # Reset simulation.
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.robot = FloatingBarret(urdf_path='../assets/barrett_gripper/floating_gripper.urdf',
                                    ee_link_name='mount_link', prefix='gripper')
        p0 = np.array([0.7, 0.0, 0.5])
        Q0 = Quaternion.from_rotation_matrix(rot_y(-np.pi))
        self.robot.reset_task_pose(p0, Q0.as_vector("wxyz"))

        # Load plane and workspace.
        pybullet_utils.load_urdf(p, PLANE_URDF_PATH, [0, 0, -0.01])
        table_id = pybullet_utils.load_urdf(p, UR5_WORKSPACE_URDF_PATH)
        p.changeDynamics(table_id, -1, lateralFriction=0.1)

        pybullet_utils.load_urdf(p, '../assets/container.urdf', [0.7, 0, 0.1],
                                 Quaternion.from_rotation_matrix(rot_z(np.pi / 2)).as_vector('xyzw'))

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Generate a scene with randomly placed objects.
        self.objects = []
        self.add_objects()

        t = 0
        while t < 1:
            self.robot.still()
            t += self.dt
            p.stepSimulation()

            time.sleep(self.dt)

        # Remove flat objects.
        self.remove_flats()

        return self.get_obs()

    def step(self, action):

        from gog.utils.pybullet_utils import draw_pose
        draw_pose(action[0]['pos'], action[0]['quat'])

        # # 1. Move the robot 10cm up the pre-grasp pose
        # approach_dir = action[0]['quat'].rotation_matrix()[:, 2]
        # pre_grasp_pos = action[0]['pos'] - approach_dir * 0.1
        pre_grasp_pos = action[0]['pos']
        pre_grasp_pos[2] += 0.1
        self.robot.move(pre_grasp_pos, action[0]['quat'], duration=5)

        # 2. Close fingers to c-shape configuration
        self.robot.move_fingers(action[0]['finger_joints'], duration=2)

        # 3. Move robot to pre-grasp position
        # self.robot.move(action[0]['pos'], action[0]['quat'], duration=5)

        draw_pose(action[1]['pos'], action[1]['quat'])
        self.robot.move(action[1]['pos'], action[1]['quat'], duration=5)
        self.robot.move_fingers(action[1]['finger_joints'], duration=2)

        final_finger_joints = action[1]['finger_joints']
        final_finger_joints[1:] += 0.2
        self.robot.move_fingers(final_finger_joints, duration=2)

        final_pos = action[1]['pos']
        final_pos[2] += 0.2
        self.robot.move(final_pos, action[1]['quat'], duration=5)

        self.robot.move(final_pos, action[1]['quat'], duration=5)

        # Move home and drop object to the bin
        self.robot.move(np.array([0.7, 0.0, 0.5]), action[1]['quat'], duration=5)
        self.robot.move_fingers([0, 0, 0, 0], duration=2)
        self.robot.move(np.array([0.7, 0.0, 0.5]), Quaternion.from_rotation_matrix(rot_y(-np.pi)), duration=5)

        return self.get_obs()

    def get_obs(self):
        """
        Get observation.
        """
        obs = {'color': [], 'depth': [], 'seg': [], 'full_state': []}

        for cam in self.agent_cams:
            color, depth, seg = cam.get_data()
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            obs['seg'] += (seg,)

        # Update position and orientation
        tmp_objects = []
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos = pos
            obj.quat = quat

            if obj.pos[2] > 0:
                tmp_objects.append(obj)

        self.objects = tmp_objects.copy()
        obs['full_state'] = self.objects

        return obs

    def remove_flats(self):
        non_flats = []
        for obj in self.objects:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)

            rot_mat = Quaternion(x=obj_quat[0], y=obj_quat[1], z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

            if obj_pos[2] < 0 or np.abs(angle_z) > 0.3:
                p.removeBody(obj.body_id)
                continue
            non_flats.append(obj)

        self.objects = non_flats

    def is_static(self):
        """
        Checks if the objects are still moving
        """
        for obj in self.objects:

            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            if pos[2] < 0:
                continue

            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm_1 = np.linalg.norm(vel)
            norm_2 = np.linalg.norm(rot_vel)
            if norm_1 > 0.001 or norm_2 > 0.1:
                return True
        return False