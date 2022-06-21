import numpy as np
import os
import math
import open3d as o3d

from gog.utils.o3d_tools import PinholeCameraIntrinsics, PointCloud


def filter(point_cloud):
    pts = []
    for pt in np.asarray(point_cloud.points):
        if np.linalg.norm(pt) != 0:
            pts.append(pt)

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def generate_views(elevation_limits, azimuth_limits, radius, resolution):
    elevations = np.arange(start=elevation_limits[0], stop=elevation_limits[1],
                           step=(elevation_limits[1] - elevation_limits[0]) / resolution)

    azimuths = np.arange(start=azimuth_limits[0], stop=azimuth_limits[1],
                         step=(azimuth_limits[1] - azimuth_limits[0]) / resolution)

    views = []
    elevation = np.pi/4
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


def collect_pcd_pairs(object_path, log_dir):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    cam_poses = generate_views(azimuth_limits=[0, 2 * np.pi],
                               elevation_limits=[np.pi / 6, np.pi / 4],
                               radius=0.5,
                               resolution=5)

    intrinsics_params = {'fx': 450, 'fy': 450, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}
    camera_intrinsics = PinholeCameraIntrinsics.from_params(intrinsics_params)

    for obj_file in os.listdir(object_path):
        print(obj_file)

        full_pcd = o3d.geometry.PointCloud()

        mesh_frames = []
        for cam_pose in cam_poses:
            model = o3d.io.read_triangle_mesh(os.path.join(object_path, obj_file))
            material = o3d.visualization.rendering.MaterialRecord()

            # print(cam_pose)
            # print(np.linalg.inv(cam_pose))
            render = o3d.visualization.rendering.OffscreenRenderer(640, 480)

            render.scene.camera.set_projection(camera_intrinsics.get_intrinsic_matrix(), 0.1, 10, 640, 480)

            render.scene.set_background([0, 0, 0, 0])
            render.scene.add_geometry("model", model, material)
            render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

            center = [0, 0, 0]  # look_at target
            eye = cam_pose[0:3, 3]  # camera position
            up = -cam_pose[0:3, 1]  # camera orientation
            render.scene.camera.look_at(center, eye, up)

            # Read the image into a variable
            # img_o3d = render.render_to_image()
            depth_o3d = render.render_to_depth_image(z_in_view_space=True)
            depth_as_img = np.array(depth_o3d)
            depth_as_img[depth_as_img == math.inf] = 0

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(img_o3d)
            # ax[1].imshow(depth_as_img)
            # plt.show()

            pcd = PointCloud.from_depth(depth_as_img, camera_intrinsics)
            pcd = filter(pcd)
            pcd.estimate_normals(fast_normal_computation=True)

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            obj_frame.transform(np.linalg.inv(cam_pose))
            model.transform(np.linalg.inv(cam_pose))

            pcd.transform(cam_pose)
            pcd.orient_normals_to_align_with_direction(-cam_pose[0:3, 2])

            mesh_frame.transform(cam_pose)
            mesh_frames.append(mesh_frame)

            full_pcd += pcd

        model = o3d.io.read_triangle_mesh(os.path.join(object_path, obj_file))
        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        mesh_frames.append(obj_frame)
        mesh_frames.append(model)
        mesh_frames.append(full_pcd)
        o3d.visualization.draw_geometries(mesh_frames)


if __name__ == "__main__":
    params = {}
    collect_pcd_pairs(object_path='../assets/objects/seen', log_dir='../logs/pcd_pairs')