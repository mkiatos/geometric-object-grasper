import numpy as np
import random
import open3d as o3d
from sklearn.neighbors import KDTree

from gog.utils.orientation import Quaternion, angle_axis2rot


def to_matrix(vec):
    pose = np.eye(4)
    pose[0:3, 0:3] = Quaternion().from_euler_angles(roll=vec[3], pitch=vec[4], yaw=vec[5]).rotation_matrix()
    pose[0:3, 3] = vec[0:3]
    return pose


class QualityMetric:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class ShapeComplementarityMetric(QualityMetric):
    def __init__(self):
        self.w = 0.5
        self.a = 0.5

    def compute_shape_complementarity(self, hand, target, radius=0.03):
        # Find nearest neighbors for each hand contact
        kdtree = KDTree(target['points'])
        dists, nn_ids = kdtree.query(hand['points'])

        shape_metric = 0.0
        for i in range(hand['points'].shape[0]):

            # Remove duplicates or contacts with nn distance greater than a threshold
            duplicates = np.argwhere(nn_ids == nn_ids[i])[:, 0]
            if np.min(dists[duplicates]) != dists[i] or dists[i] > radius:
                continue

            # Distance error
            e_p = np.linalg.norm(hand['points'][i] - target['points'][nn_ids[i][0]])

            # Alignment error
            c_n = -hand['normals'][i] # for TRO is -hand['normals']
            e_n = c_n.dot(target['normals'][nn_ids[i][0]])

            shape_metric += e_p + self.w * e_n

        return -shape_metric / len(hand['points'])

    @staticmethod
    def compute_collision_penalty(hand, collisions):
        if collisions.shape[0] == 0:
            return 0.0

        # Find nearest neighbors for each collision
        kdtree = KDTree(hand['points'])
        dists, nn_ids = kdtree.query(collisions)

        e_col = 0.0
        for i in range(collisions.shape[0]):
            e_col += np.linalg.norm(collisions[i] - hand['points'][nn_ids[i][0]])
        return e_col

    def __call__(self, hand, target, collisions):
        return self.compute_shape_complementarity(hand, target) + \
               self.a * self.compute_collision_penalty(hand, collisions)


class Optimizer:
    def __init__(self, params, name):
        self.params = params
        self.name = name

    def optimize(self):
        raise NotImplementedError


class Swarm(list):
    def __init__(self, positions, limits=None):
        super(Swarm, self).__init__()
        for i in range(positions.shape[0]):
            particle = {'pos': positions[i],
                        'prev_best_pos': positions[i],
                        'vel': 0.0,
                        'score': 0.0,
                        'prev_best_score': 0.0}
            self.append(particle)

        self.limits = limits

        self.global_best_score = 1e+308
        self.global_best_pos = self[0]['pos']

    def update(self, w, c1, c2):
        for particle in self:

            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            particle['vel'] = w * particle['vel'] + c1 * r1 * (particle['prev_best_pos'] - particle['pos']) + \
                              c2 * r2 * (self.global_best_pos - particle['pos'])
            particle['pos'] += particle['vel']

            # Bound constraints are enforced by projecting to the limits
            if self.limits is not None:
                for i in range(particle['pos'].shape[0]):
                    particle['pos'][i] = max(particle['pos'][i], self.limits[i, 0])
                    particle['pos'][i] = min(particle['pos'][i], self.limits[i, 1])

    def update_global_best(self, particle):
        # Update each particle's optimal position
        if particle['score'] < particle['prev_best_score']:
            particle['prev_best_pos'] = particle['pos']
            particle['prev_best_score'] = particle['score']

        # Update swarm's best global position
        if particle['score'] < self.global_best_score:
            self.global_best_score = particle['score']
            self.global_best_pos = particle['pos'].copy()


class SplitPSO(Optimizer):
    def __init__(self, robot, params, name='split-pso'):
        super(SplitPSO, self).__init__(name, params)
        self.robot = robot
        self.params = params

        if self.params['metric'] == 'shape_complementarity':
            self.quality_metric = ShapeComplementarityMetric()

        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng.seed(seed)

    def generate_random_angles(self, init_joints, joints_range=None):
        joints_range = np.array([[0.0, 10.0 * np.pi / 180],
                                 [0.0, 50.0 * np.pi / 180],
                                 [0.0, 50.0 * np.pi / 180],
                                 [0.0, 50.0 * np.pi / 180]])

        nr_joints = len(init_joints)

        joints_particles = np.zeros((self.params['particles'] + 1, nr_joints))
        joints_particles[0] = init_joints

        for i in range(1, self.params['particles']):
            for j in range(nr_joints):
                joints_particles[i, j] = init_joints[j] + random.uniform(joints_range[j, 0], joints_range[j, 1])

        return joints_particles

    def generate_random_poses(self, init_pose, t_range=[-0.02, 0.02], ea_range=[-20, 20]):

        pose_particles = np.zeros((self.params['particles'] + 1, 6))
        pose_particles[0, 0:3] = init_pose[0:3, 3]
        pose_particles[0, 3:] = Quaternion().from_rotation_matrix(init_pose[0:3, 0:3]).euler_angles()

        for i in range(1, self.params['particles']):
            # Randomize position
            t = init_pose[0:3, 3] + np.random.uniform(low=t_range[0], high=t_range[1], size=(3,))

            # Randomize orientation
            random_axis = np.random.rand(3, )
            random_axis /= np.linalg.norm(random_axis)
            random_rot = angle_axis2rot(np.random.uniform(ea_range[0], ea_range[1]) * np.pi / 180, random_axis)
            random_rot = np.matmul(random_rot, init_pose[0:3, 0:3])

            pose_particles[i, 0:3] = t
            pose_particles[i, 3:] = Quaternion().from_rotation_matrix(random_rot).euler_angles()

        return pose_particles

    def create_swarm(self, init_preshape):
        # Generate two swarms, one by randomizing the pose and one by randomizing the joints
        pose_swarm = Swarm(self.generate_random_poses(init_preshape['pose']))

        # ToDo: hardcoded limits
        joints_swarm = Swarm(self.generate_random_angles(init_preshape['joint_values']),
                             limits=np.array([[0, 30 * np.pi / 180], [0, 140 * np.pi / 180],
                                              [0, 140 * np.pi / 180], [0, 140 * np.pi / 180]]))

        return pose_swarm, joints_swarm

    def plot_grasp(self, point_cloud, preshape):
        self.robot.update_links(joint_values=preshape['joints'],
                                palm_pose=preshape['palm_pose'])

        hand_contacts = self.robot.get_contacts()
        object_pts = {'points': np.asarray(point_cloud.points),
                      'normals': np.asarray(point_cloud.normals)}
        collision_pts = self.robot.get_collision_pts(point_cloud)
        print('E_shape:{}, E_col:{}, E:{}'.format(self.quality_metric.compute_shape_complementarity(hand_contacts,
                                                                                                    object_pts),
                                                  self.quality_metric.compute_collision_penalty(hand_contacts,
                                                                                                collision_pts),
                                                  self.quality_metric(hand=self.robot.get_contacts(),
                                                                      target=object_pts,
                                                                      collisions=collision_pts)))
        # print('-----------')

        visuals = self.robot.get_visual_map(meshes=True, boxes=True)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        visuals.append(mesh_frame)
        visuals.append(point_cloud)
        o3d.visualization.draw_geometries(visuals)

    def optimize(self, init_preshape, point_cloud, plot=False):
        if plot:
            self.plot_grasp(point_cloud, preshape={'joints': init_preshape['joint_values'],
                                                   'palm_pose': init_preshape['pose']})

        # Generate swarms
        pose_swarm, joints_swarm = self.create_swarm(init_preshape)

        object_pts = {'points': np.asarray(point_cloud.points),
                      'normals': np.asarray(point_cloud.normals)}

        for i in range(self.params['max_iterations']):

            # # Optimize palm pose
            for particle in pose_swarm:

                # Update robot hand links
                pose = to_matrix(particle['pos'])
                self.robot.update_links(joint_values=joints_swarm.global_best_pos, palm_pose=pose)

                # Estimate each particle's fitness value(quality metric)
                particle['score'] = self.quality_metric(hand=self.robot.get_contacts(),
                                                        target=object_pts,
                                                        collisions=self.robot.get_collision_pts(point_cloud))

                # Update personal best and global best
                pose_swarm.update_global_best(particle)

            # Update velocity and position
            pose_swarm.update(w=self.params['w'], c1=self.params['c1'], c2=self.params['c2'])

            # Optimize joints
            for particle in joints_swarm:

                # Update robot hand links
                self.robot.update_links(joint_values=particle['pos'],
                                        palm_pose=to_matrix(pose_swarm.global_best_pos))

                # Estimate each particle's fitness value(quality metric)
                particle['score'] = self.quality_metric(hand=self.robot.get_contacts(),
                                                        target=object_pts,
                                                        collisions=self.robot.get_collision_pts(point_cloud))

                joints_swarm.update_global_best(particle)

            # Update velocity and position
            joints_swarm.update(w=self.params['w'], c1=self.params['c1'], c2=self.params['c2'])

            print('{}. e: {}'.format(i, joints_swarm.global_best_score))
            # self.plot_grasp(point_cloud, preshape={'joints': joints_swarm.global_best_pos,
            #                                        'palm_pose': to_matrix(pose_swarm.global_best_pos)})

        if plot:
            self.plot_grasp(point_cloud, preshape={'joints': joints_swarm.global_best_pos,
                                                   'palm_pose': to_matrix(pose_swarm.global_best_pos)})

        return {'joints': joints_swarm.global_best_pos,
                'pose': to_matrix(pose_swarm.global_best_pos),
                'score': joints_swarm.global_best_score}


class PSO(Optimizer):
    def __init__(self, robot, params, name='pso'):
        super(PSO, self).__init__(name, params)
        self.robot = robot
        self.params = params

    def optimize(self, init_preshape, point_cloud):
        pass