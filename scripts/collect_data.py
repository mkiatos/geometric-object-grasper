import numpy as np
import os
import math
import open3d as o3d
import yaml
import pickle
import argparse
import random
import shutil

from gog.cameras import RealSense
from gog.robot_hand import BarrettHand
from gog.grasp_sampler import GraspSampler
from gog.utils.o3d_tools import PinholeCameraIntrinsics, PointCloud, VoxelGrid, VoxelGrid2
from gog.utils.utils import Logger


def generate_views(elevation, azimuth_limits, radius, resolution):
    # elevations = np.arange(start=elevation_limits[0], stop=elevation_limits[1],
    #                        step=(elevation_limits[1] - elevation_limits[0]) / resolution)

    azimuths = np.arange(start=azimuth_limits[0], stop=azimuth_limits[1],
                         step=(azimuth_limits[1] - azimuth_limits[0]) / resolution)

    views = []
    # elevation = np.pi/4
    # for elevation in elevations:
    for azimuth in azimuths:
        x = radius * math.sin(elevation) * math.cos(azimuth)
        y = radius * math.sin(elevation) * math.sin(azimuth)
        z = radius * math.cos(elevation)

        r = np.array([x, y, z])
        zc = -r / np.linalg.norm(r)

        xc = np.cross(zc, np.array([0, 0, 1]))
        xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)

        view = np.eye(4)
        view[0:3, 0] = xc
        view[0:3, 1] = yc
        view[0:3, 2] = zc
        view[0:3, 3] = r

        views.append(view)

    return views


def collect_pcd_pairs(object_path, log_dir, plot=False):
    def filter(point_cloud):
        pts = []
        for pt in np.asarray(point_cloud.points):
            if np.linalg.norm(pt) != 0:
                pts.append(pt)

        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # Create the log directory
    logger = Logger(log_dir)

    cam_poses = generate_views(azimuth_limits=[0, 2 * np.pi], elevation=np.pi / 4, radius=0.5, resolution=5)

    camera_config = RealSense.CONFIG[0]
    camera_intrinsics = PinholeCameraIntrinsics.set_projection(intrinsics_matrix=np.array(camera_config['intrinsics']).reshape(3, 3),
                                                               image_size=camera_config['image_size'],
                                                               z_range=camera_config['zrange'])

    for obj_file in os.listdir(object_path):

        folder_name = os.path.join(log_dir, obj_file.split('.')[0])
        os.mkdir(folder_name)

        full_pcd = o3d.geometry.PointCloud()

        visuals = []
        counter = 0
        for cam_pose in cam_poses:
            # Load model
            model = o3d.io.read_triangle_mesh(os.path.join(object_path, obj_file))
            material = o3d.visualization.rendering.MaterialRecord()

            # Create off-screen renderer
            render = o3d.visualization.rendering.OffscreenRenderer(width=camera_intrinsics.width,
                                                                   height=camera_intrinsics.height)

            # Setup params for virtual camera
            render.scene.camera.set_projection(camera_intrinsics.get_intrinsic_matrix(), camera_intrinsics.z_near,
                                               camera_intrinsics.z_far, camera_intrinsics.width,
                                               camera_intrinsics.height)

            # Setup background and lightning
            render.scene.set_background([0, 0, 0, 0])
            render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

            # Load model
            render.scene.add_geometry("model", model, material)

            # Move camera
            center = [0, 0, 0]  # look_at target
            eye = cam_pose[0:3, 3]  # camera position
            up = -cam_pose[0:3, 1]  # camera orientation
            render.scene.camera.look_at(center, eye, up)

            # Read the image into a variable
            # img_o3d = render.render_to_image()

            # Get depth image from depth buffer
            depth_o3d = render.render_to_depth_image(z_in_view_space=True)
            depth = np.array(depth_o3d)
            depth[depth == math.inf] = 0

            # Get point cloud and remove points with zero norm
            pcd = PointCloud.from_depth(depth, camera_intrinsics)
            pcd = filter(pcd)
            pcd.estimate_normals(fast_normal_computation=True)

            if plot:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                obj_frame.transform(np.linalg.inv(cam_pose))
                model.transform(np.linalg.inv(cam_pose))

                o3d.visualization.draw_geometries([mesh_frame, obj_frame, pcd, model])

                mesh_frame.transform(cam_pose)
                visuals.append(mesh_frame)

            # Transform point cloud w.r.t. camera pose
            pcd.transform(cam_pose)
            pcd.orient_normals_to_align_with_direction(-cam_pose[0:3, 2])

            # Save partial point cloud
            o3d.io.write_point_cloud(os.path.join(folder_name, 'partial_pcd_' + str(counter) + '.pcd'), pcd)

            # Add partial cloud to create the full point cloud
            full_pcd += pcd

            counter += 1

        # Save the full point cloud
        o3d.io.write_point_cloud(os.path.join(folder_name, 'full_pcd.pcd'), full_pcd)

        print('Data generated for object {}'.format(obj_file.split('.')[0]))

        if plot:
            # model = o3d.io.read_triangle_mesh(os.path.join(object_path, obj_file))
            obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            visuals.append(obj_frame)
            # mesh_frames.append(model)
            visuals.append(full_pcd)
            o3d.visualization.draw_geometries(visuals)


def generate_local_pcd_pairs(args):
    logger = Logger(args.log_dir)

    robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    params['grasp_sampling']['samples'] = args.n_samples
    params['grasp_sampling']['rotations'] = 4

    grasp_sampler = GraspSampler(robot=robot_hand, params=params['grasp_sampling'])
    grasp_sampler.seed(args.seed)

    obj_dirs = next(os.walk(args.pcd_dir))[1]

    for obj_dir in obj_dirs:
        print('Object:', obj_dir)
        partial_clouds = []
        for pcd_file in os.listdir(os.path.join(args.pcd_dir, obj_dir)):
            if pcd_file.split('.')[0].split('_')[0] == 'partial':
                pcd = o3d.io.read_point_cloud(os.path.join(args.pcd_dir, obj_dir, pcd_file))
                pcd.paint_uniform_color(np.array([1, 0, 0]))
                partial_clouds.append(pcd)
            else:
                full_pcd = o3d.io.read_point_cloud(os.path.join(args.pcd_dir, obj_dir, pcd_file))
                full_pcd.paint_uniform_color(np.array([0, 0, 1]))

        counter = 0
        os.mkdir(os.path.join(logger.log_dir, obj_dir))
        for partial_cloud in partial_clouds:
            # print(len(partial_cloud.points))
            candidates = grasp_sampler.sample(point_cloud=partial_cloud, full_point_cloud=full_pcd, plot=False)
            for candidate in candidates:
                partial_pts = candidate['enclosing_pts'][0]
                full_pts = candidate['enclosing_pts'][1]

                # print(len(np.asarray(partial_pts.points)))
                if len(np.asarray(partial_pts.points)) < params['grasp_sampling']['minimum_pts_in_closing_region']:
                    continue

                voxel_grid = VoxelGrid(resolution=32)
                partial_grid = voxel_grid.voxelize(np.asarray(partial_pts.points), np.asarray(partial_pts.normals))
                leaf_size = voxel_grid.leaf_size
                min_pt = voxel_grid.min_pt

                voxel_grid = VoxelGrid(resolution=32)
                full_grid = voxel_grid.voxelize(np.asarray(full_pts.points), np.asarray(full_pts.normals),
                                                leaf_size, min_pt)

                # voxel_grid = VoxelGrid2()
                # voxelized_partial_cloud = voxel_grid.create_from_point_cloud(partial_pts, resolution=32)
                # partial_grid = voxel_grid.get_fused_volume()
                #
                # voxelized_full_cloud = voxel_grid.create_from_point_cloud_within_bounds(full_pts, voxel_size=,
                #                                                                         min_bound=, max_bound=)

                grid_pair = {'partial': partial_grid, 'full': full_grid, 'leaf_size': leaf_size, 'min_pt': min_pt}
                pickle.dump(grid_pair, open(os.path.join(logger.log_dir, obj_dir, 'pair_' + str(counter)), 'wb'))

                # voxelized_partial_cloud = voxel_grid.voxelized_pcd
                # voxelized_partial_cloud.paint_uniform_color(np.array([1, 0, 0]))
                # voxelized_full_cloud = voxel_grid.voxelized_pcd
                # voxelized_full_cloud.paint_uniform_color(np.array([0, 0, 1]))
                #
                # o3d.visualization.draw_geometries([full_pts, partial_pts])
                # o3d.visualization.draw_geometries([voxelized_full_cloud, voxelized_partial_cloud])
                counter += 1


def merge_folders(directories, log_dir, split_ratio=0.9):

    Logger(log_dir)
    train_logger = Logger(os.path.join(log_dir, 'train'))
    val_logger = Logger(os.path.join(log_dir, 'val'))

    random.seed(0)

    train_counter = 0
    val_counter = 0
    for directory in directories:
        obj_dirs = [x[0] for x in os.walk(directory)][1:]
        for obj_dir in obj_dirs:
            files = os.listdir(obj_dir)
            random.shuffle(files)

            train_files = files[:int(len(files) * split_ratio)]
            val_files = files[int(len(files) * split_ratio):]

            for file in train_files:
                shutil.copyfile(os.path.join(obj_dir, file),
                                os.path.join(train_logger.log_dir, 'pair_' + str(train_counter)))
                train_counter += 1

            for file in val_files:
                shutil.copyfile(os.path.join(obj_dir, file),
                                os.path.join(val_logger.log_dir, 'pair_' + str(val_counter)))
                val_counter += 1


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--n_samples', default=100, type=int, help='')
    parser.add_argument('--pcd_dir', default='../logs/pcd_pairs', type=str, help='')
    parser.add_argument('--log_dir', default='../logs/grid_pairs', type=str, help='')
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    # generate_local_pcd_pairs(args)

    dirs = ['../logs/grid_pairs_0',
            '../logs/grid_pairs_1',
            '../logs/grid_pairs_2',
            '../logs/grid_pairs_3',
            '../logs/grid_pairs_4',
            '../logs/grid_pairs_5']
    log_dir = '../logs/grid_pairs'
    merge_folders(dirs, log_dir)
