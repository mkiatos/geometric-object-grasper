import numpy as np
import itertools
import argparse
import os
import torch
from torch.utils import data
import pickle
import open3d as o3d
import random

from gog.shape_completion import ShapeCompletionNetwork
from gog.utils.o3d_tools import get_reconstructed_surface


def cross_entropy_loss(y_true, y_pred):
    x_d = y_true.shape[0]
    y_d = y_true.shape[1]
    z_d = y_true.shape[2]

    ce = - (1 / float(x_d*y_d*z_d)) * np.sum(y_true * np.log(y_pred + 1e-8) +
                                             (1 - y_true) * np.log(1 - y_pred + 1e-8))
    return ce


def jaccard_similarity(y_true, y_pred):
    """
    Compute intersection over union
    """
    y_pred = np.array(y_pred > 0.5, dtype=np.float32)
    intersection = np.sum(y_pred * y_true == 1)
    union = np.sum((y_true) >= 1)

    iou = float(intersection) / float(union)
    return iou


def cosine_similarity(y_true, y_pred, y_occupied_true):
    """
    Compute cosine similarity only for occupied voxels
    """
    x_d = y_true.shape[1]
    y_d = y_true.shape[2]
    z_d = y_true.shape[3]

    cos_sim = 0
    no_of_occupied_voxels = np.argwhere(y_occupied_true > 0.5).shape[0]
    for x, y, z in itertools.product(*map(range, (x_d, y_d, z_d))):
        if y_occupied_true[x, y, z] == 1.0:
            # print(y_true[:, x, y, z], y_pred[:, x, y, z])
            # input('')
            # y_1 = np.cos(y_true[:, x, y, z] * np.pi) / np.linalg.norm(np.cos(y_true[:, x, y, z] * np.pi))
            # y_2 = np.cos(y_pred[:, x, y, z] * np.pi) / np.linalg.norm(np.cos(y_pred[:, x, y, z] * np.pi))
            # y_1 = anges_to_normals(y_true[:, x, y, z])
            # y_2 = anges_to_normals(y_pred[:, x, y, z])

            y_1 = y_true[:, x, y, z]
            y_2 = y_pred[:, x, y, z]

            if np.sum(y_1 * y_2) > 1:
                dot = 1.0
            elif np.sum(y_1 * y_2) < -1:
                dot = -1
            else:
                dot = np.sum(y_1 * y_2)
            cos_sim += dot

    return cos_sim / no_of_occupied_voxels


def normalize(grid):
    diagonal = np.sqrt(2) * grid.shape[1]

    normalized_grid = grid.copy()
    normalized_grid[0] /= diagonal
    normalized_grid[1] = normalized_grid[1]
    normalized_grid[2:] = np.arccos(grid[2:]) / np.pi
    return normalized_grid


def remove_outliers(occupancy):
    resolution = occupancy.shape[0]
    processed_occupancy = occupancy.copy()
    for x, y, z in itertools.product(*map(range, (resolution, resolution, resolution))):
        if occupancy[x, y, z] == 1.0:

            local_grid = occupancy[max(0, x-1):min(resolution, x+2),
                                   max(0, y-1):min(resolution, y+2),
                                   max(0, z-1):min(resolution, z+2)]

            print(len(np.argwhere(local_grid == 1)))
            if len(np.argwhere(local_grid == 1)) == 1:
                print('all zeros')
                processed_occupancy[x, y, z] = 0
    return processed_occupancy


def evaluate(args):
    # grid_pair_files = os.listdir(args.test_data_dir)
    #
    # random.seed(0)
    # random.shuffle(grid_pair_files)
    # train_pairs = grid_pair_files[:int(0.9 * len(grid_pair_files))]
    # val_pairs = grid_pair_files[int(0.9 * len(grid_pair_files)):]
    # train_pairs = train_pairs[::16]
    # val_pairs = val_pairs[::16]
    random.seed(0)
    val_pairs = os.listdir(args.test_data_dir)
    random.shuffle(val_pairs)

    model = ShapeCompletionNetwork(input_dim=[5, 32, 32, 32], latent_dim=512).to('cuda')
    model.load_state_dict(torch.load(args.snapshot_file))
    model.eval()

    for grid_pair_file in val_pairs:
        print(grid_pair_file)
        grid_pair = pickle.load(open(os.path.join(args.test_data_dir, grid_pair_file), 'rb'))
        grid_pair['partial'] = normalize(grid_pair['partial'])
        grid_pair['full'] = normalize(grid_pair['full'])

        x = torch.FloatTensor(grid_pair['partial']).to('cuda')
        x = x.unsqueeze(0)

        partial_pts, partial_normals = get_reconstructed_surface(pred_sdf=grid_pair['partial'][0],
                                                                 pred_normals=grid_pair['partial'][2:],
                                                                 leaf_size=grid_pair['leaf_size'],
                                                                 min_pt=grid_pair['min_pt'])
        partial_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_pts))
        partial_point_cloud.normals = o3d.utility.Vector3dVector(partial_normals)
        partial_point_cloud.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([partial_point_cloud])

        print(np.asarray(partial_point_cloud.normals))

        pred_df, pred_normals, _, _ = model(x)

        pts, normals = get_reconstructed_surface(pred_sdf=pred_df.squeeze().detach().cpu().numpy(),
                                                 pred_normals=pred_normals.squeeze().detach().cpu().numpy(),
                                                 x=x.squeeze().detach().cpu().numpy(),
                                                 leaf_size=grid_pair['leaf_size'],
                                                 min_pt=grid_pair['min_pt'])

        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        point_cloud.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([point_cloud])

        full_pts, full_normals = get_reconstructed_surface(pred_sdf=grid_pair['full'][0],
                                                           pred_normals=grid_pair['full'][2:],
                                                           leaf_size=grid_pair['leaf_size'],
                                                           min_pt=grid_pair['min_pt'])

        pred_sdf = pred_df.squeeze(0).detach().cpu().numpy()
        pred_occupancy = np.zeros(pred_sdf.shape, dtype=np.float32)
        pred_occupancy[np.abs(pred_sdf) < 0.02] = 1.0

        gt_df = grid_pair['full'][0]
        gt_occupancy = np.zeros(gt_df.shape, dtype=np.float32)
        gt_occupancy[np.abs(gt_df) < 0.02] = 1.0

        print(jaccard_similarity(gt_occupancy, pred_occupancy))
        print('l-df:', np.mean(np.abs(pred_sdf) - gt_df) * (np.sqrt(2) * 32))
        print(len(pts), len(full_pts))

        full_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_pts))
        full_point_cloud.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([point_cloud, full_point_cloud])


def compute_metrics(args):
    # grid_pair_files = os.listdir(args.test_data_dir)
    #
    # random.seed(0)
    # random.shuffle(grid_pair_files)
    # train_pairs = grid_pair_files[:int(0.9 * len(grid_pair_files))]
    # val_pairs = grid_pair_files[int(0.9 * len(grid_pair_files)):]
    # val_pairs = val_pairs[::2]

    val_pairs = os.listdir(args.test_data_dir)
    print(len(val_pairs))

    model = ShapeCompletionNetwork(input_dim=[5, 32, 32, 32], latent_dim=512).to('cuda')
    models_path = args.model_weights
    models = []
    for model_weights in os.listdir(models_path):
        models.append(model_weights)
    models.sort()

    for model_weights in models:
        if model_weights == 'model_0.pt':
            continue
        model.load_state_dict(torch.load(os.path.join(models_path, model_weights)))
        model.eval()

        iou = 0
        for grid_pair_file in val_pairs:
            grid_pair = pickle.load(open(os.path.join(args.test_data_dir, grid_pair_file), 'rb'))

            grid_pair['partial'] = normalize(grid_pair['partial'])
            grid_pair['full'] = normalize(grid_pair['full'])

            x = torch.FloatTensor(grid_pair['partial']).to('cuda')
            x = x.unsqueeze(0)

            pred_sdf, pred_normals, _, _ = model(x)

            pred_sdf = pred_sdf.squeeze(0).detach().cpu().numpy()
            pred_occupancy = np.zeros(pred_sdf.shape, dtype=np.float32)
            pred_occupancy[np.abs(pred_sdf) < 0.02] = 1.0

            gt_df = grid_pair['full'][0]
            gt_occupancy = np.zeros(gt_df.shape, dtype=np.float32)
            gt_occupancy[np.abs(gt_df) < 0.02] = 1.0

            iou += jaccard_similarity(gt_occupancy, pred_occupancy)
            # print(jaccard_similarity(gt_occupancy, pred_occupancy))

        print('model:{}, iou:{}'.format(model_weights, iou/len(val_pairs)))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_weights', default='', type=str, help='')
    parser.add_argument('--test_data_dir', default='', type=str, help='')
    parser.add_argument('--snapshot_file', default='', type=str, help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # compute_metrics(args)
    evaluate(args)
