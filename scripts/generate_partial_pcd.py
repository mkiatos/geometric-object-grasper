import numpy as np
import open3d as o3d
import os
import math
import argparse

from gog.cameras import RealSense
from gog.utils.o3d_tools import PinholeCameraIntrinsics, PointCloud
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


def collect_pcd_pairs(args):
    def filter(point_cloud):
        pts = []
        for pt in np.asarray(point_cloud.points):
            if np.linalg.norm(pt) != 0:
                pts.append(pt)

        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # Create the log directory
    logger = Logger(args.log_dir)

    cam_poses = generate_views(azimuth_limits=[0, 2 * np.pi], elevation=np.pi / 4, radius=0.5, resolution=args.n_views)

    camera_config = RealSense.CONFIG[0]
    camera_intrinsics = PinholeCameraIntrinsics.set_projection(intrinsics_matrix=np.array(camera_config['intrinsics']).reshape(3, 3),
                                                               image_size=camera_config['image_size'],
                                                               z_range=camera_config['zrange'])

    object_path = os.path.join('../assets/objects', args.object_set)
    for obj_file in os.listdir(object_path):

        folder_name = os.path.join(logger.log_dir, obj_file.split('.')[0])
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

            if args.plot:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                obj_frame.transform(np.linalg.inv(cam_pose))
                model.transform(np.linalg.inv(cam_pose))

                # o3d.visualization.draw_geometries([mesh_frame, obj_frame, pcd, model])

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

        if args.plot:
            # model = o3d.io.read_triangle_mesh(os.path.join(object_path, obj_file))
            obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            visuals.append(obj_frame)
            # mesh_frames.append(model)
            visuals.append(full_pcd)
            o3d.visualization.draw_geometries(visuals)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--object_set', default='seen', type=str, help='')
    parser.add_argument('--n_views', default=5, type=int, help='The number of views to render')
    parser.add_argument('--log_dir', default='../logs/pcd_pairs', type=str, help='')
    parser.add_argument('--plot',  action='store_true', default=False, help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_pcd_pairs(args)
