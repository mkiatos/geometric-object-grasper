import pickle

import numpy as np
import kinpy as kp
import open3d as o3d
import os
import copy
import math

from gog.utils.orientation import Quaternion
from gog.utils.utils import sample_surface

default_params = {
    'c_shape_hand': {
        'gripper_width': 0.07,
        'inner_radius': [0.05, 0.06, 0.07, 0.08],
        'outer_radius': [0.08, 0.09, 0.1, 0.11],
        'phi': [np.pi / 3, np.pi / 4, np.pi / 6, 0],
        'bhand_angles': [1.2, 1, 0.8, 0.65]
    }
}


class Link:
    """
    Class that defines a link of the robot

    Parameters
    ----------
    offset: np.array()
        1x3 array defining the offset for each axe
    geom_type: string
        The geometry type of the link i.e. 'mesh', 'box', None
    geom_param: string
        If the geom_type is mesh, this variable defines the mesh file.
    """
    def __init__(self, name, offset, geom_type=None,  geom_param=None):
        self.name = name
        self.offset = offset
        self.geom_type = geom_type

        # If the geom_type is mesh load the mesh model and get its oriented bounding box
        if geom_type == 'mesh':
            self.geom_param = geom_param
            self.mesh = o3d.io.read_triangle_mesh(geom_param)
            self.obb = self.mesh.get_oriented_bounding_box()
            self.contacts = o3d.geometry.PointCloud()
        else:
            self.geom_param = None
            self.contacts = o3d.geometry.PointCloud()

        # Create a frame at the origin of the link
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

    def transform(self, pos, rot):
        self.frame.translate(pos)
        self.frame.rotate(rot, pos)

        if self.geom_type == 'mesh':
            self.obb.translate(pos)
            self.obb.rotate(rot, pos)

            self.mesh.translate(pos)
            self.mesh.rotate(rot, pos)

            # Check if contacts are loaded
            if not self.contacts.is_empty():
                self.contacts.translate(pos)
                self.contacts.rotate(rot, pos)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Link(offset=copy.copy(self.offset), geom_type=copy.copy(self.geom_type),
                    geom_param=copy.copy(self.geom_param))

    def __str__(self):
        if self.contacts.is_empty():
            no_of_contacts = 0
        else:
            no_of_contacts = np.asarray(self.contacts).shape[0]
        return 'Link_name: {}, no_of_contacts: {}'.format(self.name, no_of_contacts)


class RobotHand:
    """
    Base class for robot hand

    Parameters
    ----------
    name : string
        the name of robot hand.
    urdf_file : string
        the urdf file to load the robot hand.
    """
    def __init__(self,
                 urdf_file,
                 name):
        self.name = name
        self.assets_dir = os.path.join('../assets/robot_hands', self.name)

        # Build kinematic chain from urdf
        self.chain = kp.build_chain_from_urdf(open(urdf_file).read())
        self.visuals_map = self.chain.visuals_map()

        # Load links
        self.links = {}
        for link_name in self.visuals_map:

            v = self.visuals_map[link_name][0]

            if v.geom_type != 'mesh':
                self.links[link_name] = Link(name=link_name, offset=v.offset, geom_type=v.geom_type)
            else:
                self.links[link_name] = Link(name=link_name, offset=v.offset, geom_type=v.geom_type,
                                             geom_param=os.path.join(self.assets_dir, v.geom_param))

        # Set the reference frame of the robot hand (aligned with the c-shape ref frame) w.r.t. robot hand base frame
        self.ref_frame = np.eye(4)

        self.ignore_links = {}
        self.contacts = {}

        self.load_virtual_contacts('contacts_230')

    def forward_kinematics(self, joint_values, palm_pose):
        """
        Computes the forward kinematics given the joint values and the palm pose

        Parameters
        ----------
        joint_values: list
            the values of the actuated joints (spread, finger1, finger2, finger3)
        palm_pose: np.array()
            4x4 matrix defining the pose of the palm w.r.t. world
        """
        # Todo: change fk to compute from differences and NOT loading every time the hand

        ret = self.chain.forward_kinematics(joint_values)

        palm_trans = kp.Transform(rot=Quaternion().from_rotation_matrix(palm_pose[0:3, 0:3]).as_vector(),
                                  pos=palm_pose[0:3, 3])

        base_trans = kp.Transform(rot=Quaternion().from_rotation_matrix(self.ref_frame[0:3, 0:3]).as_vector(),
                                  pos=self.ref_frame[0:3, 3])

        for link_name, trans in ret.items():
            if link_name in self.ignore_links:
                continue
            v = self.visuals_map[link_name][0]

            # tf = palm_trans * trans * v.offset
            tf = palm_trans * base_trans * trans * v.offset
            rot = Quaternion(w=tf.rot[0], x=tf.rot[1], y=tf.rot[2], z=tf.rot[3]).rotation_matrix()

            if v.geom_type != 'mesh':
                self.links[link_name] = Link(name=link_name, offset=v.offset, geom_type=v.geom_type)
                self.links[link_name].transform(pos=tf.pos, rot=rot)
            else:
                self.links[link_name] = Link(name=link_name, offset=v.offset, geom_type=v.geom_type,
                                             geom_param=os.path.join(self.assets_dir, v.geom_param))

                if self.contacts:
                    self.links[link_name].contacts.points = o3d.utility.Vector3dVector(
                        self.contacts[link_name]['points'])
                    self.links[link_name].contacts.normals = o3d.utility.Vector3dVector(
                        self.contacts[link_name]['normals'])
                    self.links[link_name].contacts.paint_uniform_color(color=[1, 0, 0])

                self.links[link_name].transform(pos=tf.pos, rot=rot)

    def get_visual_map(self, meshes=True, boxes=False, frames=False):
        """
        Returns a visual map of the robot hand's links
        """
        visuals = []
        for key in self.links:
            if key in self.ignore_links:
                continue

            if self.links[key].geom_type == 'mesh':
                if meshes:
                    visuals.append(self.links[key].mesh)
                if boxes:
                    visuals.append(self.links[key].obb)
                if frames:
                    visuals.append(self.links[key].frame)

                visuals.append(self.links[key].contacts)
        return visuals

    def update_links(self):
        raise NotImplementedError

    def is_collision_free(self, point_cloud):
        """
        Checks for collisions of the robot hand's links with a point cloud. Each link is approximated with
        an oriented bounding box.

        Parameters
        ----------
        point_cloud: open3d.geometry.PointCloud
            The input point cloud to be checked for collisions with the hand.

        Returns
        -------
        bool: True if there is no collision else False.
        """
        for key in self.links:
            if key in self.ignore_links:
                continue
            if self.links[key].geom_type == 'mesh':
                if len(self.links[key].obb.get_point_indices_within_bounding_box(point_cloud.points)) > 0:
                    return False
        return True

    def get_collision_pts(self, point_cloud):
        collision_ids = []
        for key in self.links:
            if key in self.ignore_links:
                continue
            if self.links[key].geom_type == 'mesh':
                collision_ids += self.links[key].obb.get_point_indices_within_bounding_box(point_cloud.points)

        return np.asarray(point_cloud.select_by_index(collision_ids).points)

    def load_virtual_contacts(self, contacts_file):
        self.contacts = pickle.load(open(os.path.join(self.assets_dir, 'virtual_contacts',
                                                      contacts_file), 'rb'))
        for link_name in self.contacts:
            self.links[link_name].contacts.points = o3d.utility.Vector3dVector(self.contacts[link_name]['points'])
            self.links[link_name].contacts.normals = o3d.utility.Vector3dVector(self.contacts[link_name]['normals'])
            self.links[link_name].contacts.paint_uniform_color(color=[0, 0, 1])

    def get_joint_values(self):
        pass

    def get_pose(self):
        pass

    def get_contacts(self):
        pts = []
        normals = []
        for link_name in self.links:
            pts += np.asarray(self.links[link_name].contacts.points).tolist()
            normals += np.asarray(self.links[link_name].contacts.normals).tolist()

        return {'points': np.asarray(pts), 'normals': np.asarray(normals)}


class BarrettHand(RobotHand):
    def __init__(self,
                 urdf_file,
                 name='barrett'):
        super().__init__(urdf_file, name)

        # Define reference frame
        self.ref_frame[0:3, 3] = np.array([0.0, 0.0, 0.1]) # ToDo: hardcoded?
        self.ref_frame = np.linalg.inv(self.ref_frame)

        # Define links that will be ignored
        self.ignore_links = {'bh_finger_31_link', 'bh_finger_21_link', 'bh_finger_11_link'}
        # self.ignore_links = {}

    def update_links(self,
                     joint_values=[0.0, 0.0, 0.0, 0.0],
                     palm_pose=np.eye(4)):
        joints_dict = {'bh_j11_joint': joint_values[0], 'bh_j21_joint': joint_values[0],
                       'bh_j12_joint': joint_values[1], 'bh_j13_joint': joint_values[1] / 3,
                       'bh_j22_joint': joint_values[2], 'bh_j23_joint': joint_values[2] / 3,
                       'bh_j32_joint': joint_values[3], 'bh_j33_joint': joint_values[3] / 3}
        self.forward_kinematics(joints_dict, palm_pose)


class RobotiqHand(RobotHand):
    def __init__(self, urdf_file, name='robotiq'):
        super(RobotiqHand, self).__init__(urdf_file, name)
        self.ignore_links = {'finger_1_link_0', 'finger_2_link_0', 'finger_middle_link_0'}

        self.ref_frame[0:3, 3] = np.array([0.0, 0.0, 0.1])
        self.ref_frame = np.linalg.inv(self.ref_frame)

    def update_links(self, joint_values, palm_pose=np.eye(4)):
        joints_dict = {'finger_1_link_0': 0.0}

        self.forward_kinematics(joints_dict, palm_pose)


class CShapeHand:
    def __init__(self, inner_radius, outer_radius, gripper_width, phi):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.gripper_width = gripper_width
        self.phi = phi

        # base frame of the c_shape (placed at the center of inner radius) w.r.t. base frame of robot hand
        self.base_frame = np.eye(4)
        self.base_frame[2, 3] = self.inner_radius - 0.04

    def fit_robot_hand(self, robot_hand):
        # ToDo: This function should estimate the c_shape parameters
        pass

    def is_collision_free(self, grasp_candidate, pts):
        grasp_candidate = np.matmul(grasp_candidate, self.base_frame)

        cylinder_axis = grasp_candidate[0:3, 0]
        p = grasp_candidate[0:3, 3]

        # Compute the centers of the circular facets of cylinder
        p1 = p + cylinder_axis * self.gripper_width / 2
        p2 = p - cylinder_axis * self.gripper_width / 2

        # Compute the corners of the rectangular box (we only need 4 of them in order to
        # compute the main directions)
        # p1_box = p1 - grasp_candidate[0:3, 1] * self.inner_radius
        # p2_box = p1 + grasp_candidate[0:3, 1] * self.inner_radius
        # p3_box = p2 - grasp_candidate[0:3, 1] * self.inner_radius
        # p4_box = p1_box + grasp_candidate[0:3, 2] * self.outer_radius

        p1_box = p1 - grasp_candidate[0:3, 1] * math.cos(self.phi) * self.inner_radius
        p3_box = p1_box + grasp_candidate[0:3, 2] * self.outer_radius
        p1_box += grasp_candidate[0:3, 2] * math.sin(self.phi) * self.inner_radius
        p2_box = p1_box + grasp_candidate[0:3, 1] * math.cos(self.phi) * 2 * self.inner_radius
        p4_box = p1_box - grasp_candidate[0:3, 0] * self.gripper_width

        # Three perpendicular directions of the rectangular box
        u = p4_box - p1_box
        v = p2_box - p1_box
        w = p3_box - p1_box

        # Check for collisions
        for pt in pts:

            # Confirm that point lies between the planes of the two circular
            # facets of the cylinder
            if (pt - p1).dot(p2 - p1) >= 0 and (pt - p2).dot(p1 - p2) >= 0:

                # Confirm that the point lies inside the curved surface of the c-shape
                r = np.linalg.norm(np.cross(pt - p1, p2 - p1)) / np.linalg.norm(p2 - p1)
                if self.inner_radius < r < self.outer_radius:

                    # Check if q lies inside the rectangular region (see Lei paper).
                    # If not then q is a collision
                    uq = u.dot(pt)
                    vq = v.dot(pt)
                    wq = w.dot(pt)
                    if u.dot(p1_box) < uq < u.dot(p4_box) and \
                            v.dot(p1_box) < vq < v.dot(p2_box) and \
                            w.dot(p1_box) < wq < w.dot(p3_box):
                        continue
                    else:
                        return False

        return True

    def push_forward(self, grasp_candidate, pts):
        tmp = grasp_candidate.copy()

        push_step_size = 0.005
        push_distance = 0.0

        while True:
            push_distance += push_step_size
            tmp[0:3, 3] += tmp[0:3, 2] * push_step_size

            if not self.is_collision_free(tmp, pts):
                break

        tmp[0:3, 3] -= tmp[0:3, 2] * push_step_size
        return tmp

    def get_points_in_closing_region(self, grasp_candidate, pts):
        grasp_candidate = np.matmul(grasp_candidate, self.base_frame)
        p1 = grasp_candidate[0:3, 3] + grasp_candidate[0:3, 0] * self.gripper_width / 2
        p2 = grasp_candidate[0:3, 3] - grasp_candidate[0:3, 0] * self.gripper_width / 2

        enclosing_ids = []
        for i in range(len(pts)):
            pt = pts[i]
            # Confirm that point lies between the planes of the two circular
            # facets of the cylinder
            if (pt - p1).dot(p2 - p1) >= 0 and (pt - p2).dot(p1 - p2) >= 0:

                # Confirm that the point lies inside the curved surface of the c-shape
                r = np.linalg.norm(np.cross(pt - p1, p2 - p1)) / np.linalg.norm(p2 - p1)
                if r <= self.outer_radius:
                    enclosing_ids.append(i)

        return enclosing_ids

    def get_visual_map(self):
        return self.sample_surface()

    def sample_surface(self, point_density=0.001):
        angle_step = point_density / self.inner_radius
        y_box = math.cos(self.phi) * self.inner_radius
        outer_angle = math.acos(y_box / self.outer_radius)
        z_max = math.sin(outer_angle) * self.outer_radius

        x_values = np.arange(-self.gripper_width/2, self.gripper_width/2, point_density)
        z_values = np.arange(math.sin(self.phi) * self.inner_radius, z_max, point_density)
        angle_values = np.arange(-self.phi, np.pi + self.phi, angle_step)

        pts = []

        for x in x_values:

            for z in z_values:
                pts.append([x, y_box, z])
                pts.append([x, -y_box, z])

            for angle in angle_values:
                pts.append([x, self.inner_radius * math.cos(angle),
                            -self.inner_radius * math.sin(angle)])

        outer_angles = np.arange(-outer_angle, np.pi + outer_angle, angle_step)
        for x in x_values:
            for angle in outer_angles:
                pts.append([x, self.outer_radius * math.cos(angle),
                            -self.outer_radius * math.sin(angle)])

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pts)
        point_cloud.paint_uniform_color(np.array([.5, 0.5, 0.5]))

        point_cloud.transform(self.base_frame)
        return point_cloud

