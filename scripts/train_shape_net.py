import os
import pickle
import random
import argparse
import numpy as np

from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import torch

from gog.utils.utils import Logger
from gog.shape_completion import ShapeCompletionNetwork, ShapeCompletionLoss


class GridDataset(data.Dataset):
    def __init__(self, log_dir, grid_pairs):
        super(GridDataset, self).__init__()
        self.log_dir = log_dir
        self.grid_pairs = grid_pairs

        # if os.path.exists(os.path.join(log_dir, 'statistics')):
        #     statistics = pickle.load(open(os.path.join(log_dir, 'statistics'), 'rb'))
        #     self.mean = statistics['mean'].detach().cpu().numpy()
        #     self.std = statistics['std'].detach().cpu().numpy()
        # else:
        #     self.mean = None
        #     self.std = None

    @staticmethod
    def normalize(grid):
        diagonal = np.sqrt(2) * grid.shape[1]

        normalized_grid = grid.copy()
        normalized_grid[0] /= diagonal
        normalized_grid[1] = normalized_grid[1]
        normalized_grid[2:] = np.arccos(grid[2:]) / np.pi
        return normalized_grid

    def __getitem__(self, idx):
        grid_pair = pickle.load(open(os.path.join(self.log_dir, self.grid_pairs[idx]), 'rb'))
        # if self.mean is not None:
        #     partial = np.zeros(grid_pair['partial'].shape)
        #     for i in range(5):
        #         partial[i] = (grid_pair['partial'][i] - self.mean[i]) / self.std[i]
        # else:
        #     partial = grid_pair['partial']
        # print(np.min(grid_pair['partial']), np.max(grid_pair['partial']))
        partial_grid = self.normalize(grid_pair['partial'])
        full_grid = self.normalize(grid_pair['full'])
        # print(np.min(partial_grid), np.max(partial_grid))
        # print(np.min(full_grid), np.max(full_grid))
        return partial_grid, full_grid

    def __len__(self):
        return len(self.grid_pairs)


def train(args):
    def mask_out(y, delta=0.05):
        y = y.detach().cpu().numpy()
        weight = np.zeros(y.shape)
        weight[np.abs(y) < delta] = 1.0
        return torch.tensor(weight, dtype=torch.float, requires_grad=True).to('cuda')

    logger = Logger(log_dir='../logs/vae_2')

    # grid_pairs = os.listdir(args.dataset_dir)

    # Split data to training/validation
    # random.seed(0)
    # random.shuffle(grid_pairs)

    # train_pairs = grid_pairs[:int(args.split_ratio * len(grid_pairs))]
    # val_pairs = grid_pairs[int(args.split_ratio * len(grid_pairs)):]
    # train_pairs = train_pairs[::16]
    # val_pairs = val_pairs[::16]

    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'val')

    train_pairs = os.listdir(train_dir)
    val_pairs = os.listdir(val_dir)

    # train_pairs = train_pairs[::16]
    # val_pairs = val_pairs[::16]

    train_dataset = GridDataset(train_dir, train_pairs)
    val_dataset = GridDataset(val_dir, val_pairs)

    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)
    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_pairs), len(val_pairs)))

    model = ShapeCompletionNetwork(input_dim=[5, 32, 32, 32], latent_dim=512).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = ShapeCompletionLoss()

    for epoch in range(args.epochs):
        model.train()
        for batch in data_loader_train:
            x = batch[0].to(dtype=torch.float, device='cuda')
            # print(np.min(x.detach().cpu().numpy()), np.max(x.detach().cpu().numpy()))
            y = batch[1].to(dtype=torch.float, device='cuda')

            # Input volume 5-channels: 0-df, 1-known/unknown regions, 2-5 direction angles
            # Ground truth
            sign = y[:, 1, :, :, :].detach().cpu().numpy().copy()
            sign[sign == 1] = -1
            sign[sign == 0] = 1
            sign = torch.tensor(sign, requires_grad=True).to(dtype=torch.float, device='cuda')

            y_sdf = (y[:, 0, :, :, :] * sign).unsqueeze(1)
            # y_sdf = y[:, 0, :, :, :].unsqueeze(1)
            y_normals = y[:, 2:, :, :, :]

            mask_sdf = x[:, 1, :, :, :].unsqueeze(1)
            mask_normals = mask_out(y_normals, delta=0.05)

            # Compute loss
            pred_sdf, pred_normals, mu, logvar = model(x)
            loss = criterion(y_sdf, pred_sdf, y_normals, pred_normals, mask_sdf, mask_normals, mu, logvar)

            # Update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save model and print progress
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(logger.log_dir, 'model_' + str(epoch) + '.pt'))

            model.eval()
            epoch_loss = {'train': 0.0, 'val': 0.0}
            for phase in ['train', 'val']:
                for batch in data_loaders[phase]:
                    x = batch[0].to(dtype=torch.float, device='cuda')
                    y = batch[1].to(dtype=torch.float, device='cuda')

                    # Input volume 5-channels: 0-df, 1-known/unknown regions, 2-5 direction angles
                    # Ground truth
                    sign = y[:, 1, :, :, :].detach().cpu().numpy().copy()
                    sign[sign == 1] = -1
                    sign[sign == 0] = 1
                    sign = torch.tensor(sign, requires_grad=True).to(dtype=torch.float, device='cuda')

                    y_sdf = (y[:, 0, :, :, :] * sign).unsqueeze(1)
                    y_normals = y[:, 2:, :, :, :]

                    mask_sdf = x[:, 1, :, :, :].unsqueeze(1)
                    mask_normals = mask_out(y_normals, delta=0.05)

                    # Compute loss
                    pred_sdf, pred_normals, mu, logvar = model(x)
                    loss = criterion(y_sdf, pred_sdf, y_normals, pred_normals, mask_sdf, mask_normals, mu, logvar)

                    loss = torch.sum(loss)
                    epoch_loss[phase] += loss.detach().cpu().numpy()

            print('Epoch {}: training loss = {:.4f} '
                  ', validation loss = {:.4f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                      epoch_loss['val'] / len(data_loaders['val'])))


def compute_dataset_statistics(args):
    grid_pairs = os.listdir(args.dataset_dir)
    dataset = GridDataset(args.dataset_dir, grid_pairs)
    loader = data.DataLoader(dataset,
                             batch_size=10,
                             num_workers=0,
                             shuffle=False)

    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(mean, std)

    pickle.dump({'mean': mean, 'std': std}, open(os.path.join(args.dataset_dir, 'statistics'), 'wb'))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir', default='../logs/grid_pairs', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')
    parser.add_argument('--save_every', default=5, type=int, help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # compute_dataset_statistics(args)
    train(args)
