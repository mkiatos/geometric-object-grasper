import os


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


def test_grasp_sampler():
    from gog.robot_hand import BarrettHand
    from gog.environment import Environment
    from gog.policy import GraspingPolicy
    import yaml

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')

    env = Environment(assets_root='../assets')
    env.seed(12)
    obs = env.reset()

    policy = GraspingPolicy(robot_hand=robot_hand, params=params)
    state = policy.state_representation(obs, plot=True)
    candidates = policy.get_samples(state)

    # for candidate in candidates:
    #     print(candidate)
    #     env.step(candidate)

    # grasp_sampler = GraspSampler(robot=robot_hand, params=params['grasp_sampling'])
    # grasp_candidates = grasp_sampler.sample(obs, plot=True)


if __name__ == "__main__":
    # test_c_shape()
    test_grasp_sampler()