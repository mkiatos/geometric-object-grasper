import os
import open3d as o3d


def test_c_shape():
    import open3d as o3d
    import numpy as np
    from gog.robot_hand import BarrettHand, CShapeHand

    default_params = {
        'gripper_width': 0.07,
        'inner_radius': [0.05, 0.06, 0.07, 0.08],
        'outer_radius': [0.09, 0.1, 0.11, 0.12],
        'phi': [np.pi / 3, np.pi / 4, np.pi / 6, 0],
        'bhand_angles': [1.2, 1, 0.8, 0.65]
    }

    robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')

    for i in range(4):
        init_angle = default_params['bhand_angles'][i]
        robot_hand.update_links(joint_values=[0.0, init_angle, init_angle, init_angle])
        visuals = robot_hand.get_visual_map(frames=True)

        c_shape_hand = CShapeHand(inner_radius=default_params['inner_radius'][i],
                                  outer_radius=default_params['outer_radius'][i],
                                  gripper_width=default_params['gripper_width'],
                                  phi=default_params['phi'][i])
        point_cloud = c_shape_hand.sample_surface()
        visuals.append(point_cloud)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        visuals.append(frame)
        o3d.visualization.draw_geometries(visuals)


def test_optimizer():
    from gog.robot_hand import BarrettHand
    from gog.grasp_optimizer import SplitPSO
    from gog.policy import GraspingPolicy
    import yaml
    import open3d as o3d

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')

    params['grasp_sampling']['rotations'] = 1

    optimizer = SplitPSO(robot_hand, params['power_grasping']['optimizer'])

    policy = GraspingPolicy(robot_hand=robot_hand, params=params)

    point_cloud = o3d.io.read_point_cloud(os.path.join('../logs/pcd_pairs', '47', 'full_pcd.pcd'))
    o3d.visualization.draw_geometries([point_cloud])

    candidates = policy.get_candidates(point_cloud)

    for candidate in candidates:
        init_preshape = {'pose': candidate['pose'], 'joint_values': candidate['joint_vals']}
        optimized_preshape = optimizer.optimize(init_preshape, point_cloud)


def test_gog_planner():
    from gog.robot_hand import BarrettHand
    from gog.environment import Environment
    from gog.policy import GraspingPolicy
    import yaml
    import numpy as np

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')

    env = Environment(assets_root='../assets')
    env.seed(1)
    obs = env.reset()

    policy = GraspingPolicy(robot_hand=robot_hand, params=params)
    policy.seed(10)

    while True:
        state = policy.state_representation(obs, plot=True)

        # Sample candidates
        candidates = policy.get_candidates(state, plot=True)
        print('{} grasp candidates found...!'.format(len(candidates)))

        scores = []
        actions = []
        for candidate in candidates:
            rec_point_cloud = policy.reconstruct(candidate, plot=True)
            rec_point_cloud = rec_point_cloud.transform(candidate['pose'])

            init_preshape = {'pose': candidate['pose'], 'joint_values': candidate['joint_vals']}
            opt_preshape = policy.optimize(init_preshape, rec_point_cloud)

            action = {'pose': [init_preshape['pose'], opt_preshape['pose']],
                      'joint_vals': [candidate['joint_vals'], opt_preshape['joints']]}
            actions.append(action)
            scores.append(opt_preshape['score'])

        best_id = np.argmin(scores)
        obs = env.step(actions[best_id])


def test_floating_gripper(plot=True):
    from gog.robot_hand import BarrettHand
    from gog.environment import Environment
    from gog.policy import GraspingPolicy
    from gog.utils.orientation import Quaternion
    import yaml
    import numpy as np

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')

    env = Environment(assets_root='../assets')
    env.seed(1)
    obs = env.reset()

    policy = GraspingPolicy(robot_hand=robot_hand, params=params)
    policy.seed(10)

    while True:
        print('-----------------------')

        state = policy.state_representation(obs, plot=False)

        if len(np.where(np.asarray(state.points)[:, 2] > 0.01)[0]) < 100:
            break

        # Sample candidates
        candidates = policy.get_candidates(state, plot=False)
        print('{} grasp candidates found...!'.format(len(candidates)))

        scores = []
        actions = []
        preshapes = []
        for i in range(len(candidates)):
            text = str(i+1) + '.'
            print(text, end='\r')

            candidate = candidates[i]
            # Local reconstruction
            rec_point_cloud = policy.reconstruct(candidate, plot=False)
            rec_point_cloud = rec_point_cloud.transform(candidate['pose'])

            text += 'Local region reconstructed.'
            print(text, end='\r')

            # Grasp optimization
            init_preshape = {'pose': candidate['pose'], 'finger_joints': candidate['finger_joints']}
            opt_preshape = policy.optimize(init_preshape, rec_point_cloud, plot=True)

            text += 'Grasp optimized e: ' + str(np.abs(opt_preshape['score']))
            print(text)

            # Execute the best action i.e. the action with the highest score
            action0 = {'pos': candidate['pose'][0:3, 3],
                       'quat': Quaternion().from_rotation_matrix(candidate['pose'][0:3, 0:3]),
                       'finger_joints': candidate['finger_joints']}
            action1 = {'pos': opt_preshape['pose'][0:3, 3],
                       'quat': Quaternion().from_rotation_matrix(opt_preshape['pose'][0:3, 0:3]),
                       'finger_joints': opt_preshape['finger_joints']}
            action = [action0, action1]

            actions.append(action)
            scores.append(opt_preshape['score'])
            preshapes.append(opt_preshape)

        # Execute the preshape with the highest score
        opt_id = np.argmin(scores)

        if plot:
            pose = np.matmul(preshapes[opt_id]['pose'], np.linalg.inv(robot_hand.ref_frame))
            robot_hand.update_links(joint_values=preshapes[opt_id]['finger_joints'],
                                    palm_pose=pose)

            hand_contacts = robot_hand.get_contacts()
            pcd_contacts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hand_contacts['points']))
            pcd_contacts.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([pcd_contacts, state])

        obs = env.step(actions[opt_id])


def parse_args():
    return []


if __name__ == "__main__":
    # test_c_shape()
    # test_optimizer()
    # test_gog_planner()

    test_floating_gripper()