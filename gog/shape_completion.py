import torch
import torch.nn as nn

default_params = {
    'Encoder': {
        'channels': [64, 128, 256, 512],
        'kernels': [4, 4, 4, 4],
        'strides': [2, 2, 2, 2],
        'padding': 'same',
        'pooling': {
            'mode': 'max',
            'kernel': 2,
            'stride': 2
        }
    },
    'Decoder': {
        'channels': [[1024, 256], [512, 128], [256, 64], [128, 1], [128, 3]],
        'kernels': [4, 4, 4, 4, 4],
        'strides': [1, 2, 2, 2, 2],
        'padding': [0, 1, 1, 1, 1],
    },
    'device': 'cuda'
}


class ShapeCompletionNetwork(nn.Module):
    """
    Symmetrical Autoencoder with skip connections
    """
    def __init__(self, input_dim, latent_dim, params=default_params):
        super(ShapeCompletionNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Encoder
        encoder_params = params['Encoder']
        if encoder_params['padding'] == 'same':
            self.encoder_padding = (1, 2, 1, 2, 1, 2)

        input_channels, _, _, _ = input_dim
        encoder_params['channels'].insert(0, input_channels)

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        for i in range(len(encoder_params['channels']) - 1):
            self.conv_layers.append(nn.Conv3d(encoder_params['channels'][i],
                                              encoder_params['channels'][i+1],
                                              encoder_params['kernels'][i],
                                              encoder_params['strides'][i],
                                              padding=0))
            self.pool_layers.append(nn.MaxPool3d(encoder_params['pooling']['kernel'],
                                                 encoder_params['pooling']['stride']))
            self.batch_norm_layers.append(nn.BatchNorm3d(encoder_params['channels'][i+1],
                                                         affine=False))

        self.encoder_fc = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512)
        self.batch_norm_layers_deconv = nn.ModuleList()

        decoder_params = params['Decoder']
        self.deconv_layers = nn.ModuleList()
        for i in range(len(decoder_params['channels'])):
            self.deconv_layers.append(nn.ConvTranspose3d(decoder_params['channels'][i][0],
                                                         decoder_params['channels'][i][1],
                                                         decoder_params['kernels'][i],
                                                         decoder_params['strides'][i],
                                                         decoder_params['padding'][i]))
            self.batch_norm_layers_deconv.append(nn.BatchNorm3d(decoder_params['channels'][i][1],
                                                                affine=False))
        self.pool_output = []
        # initialize layers

        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(512, latent_dim)

        # Latent
        self.z = []

    def encode(self, x_grid):
        x = x_grid.clone().detach()
        for i in range(len(self.conv_layers) - 1):
            x = nn.functional.pad(x, self.encoder_padding, 'constant', 0)
            x = nn.functional.leaky_relu(self.conv_layers[i](x))
            # x = self.batch_norm_layers[i](x)
            # x = self.pool_layers[i](x)
            self.pool_output.append(x)

        x = nn.functional.leaky_relu(self.conv_layers[-1](x))
        self.pool_output.append(x)

        x = x.view(-1, 512 * 1 * 1 * 1)
        # x = torch.cat((x, x_pose), 1)
        z = nn.functional.relu(self.encoder_fc(x))
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        n = x.shape[0]
        x = x.view(n, 512, 1, 1, 1)

        self.pool_output = self.pool_output[::-1]
        for i in range(len(self.deconv_layers) - 2):
            x = torch.cat((x, self.pool_output[i]), dim=1)
            x = nn.functional.relu(self.deconv_layers[i](x))
            # x = self.batch_norm_layers_deconv[i](x)

        x = torch.cat((x, self.pool_output[-1]), dim=1)

        # signed distance field
        pred_df = torch.tanh(self.deconv_layers[-2](x))
        # pred_df = self.deconv_layers[-2](x)

        # normals
        pred_normals = torch.sigmoid(self.deconv_layers[-1](x))
        # pred_normals = self.deconv_layers[-1](x)

        self.pool_output.clear()
        return pred_df, pred_normals

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred_occupancy, pred_normals = self.decode(z)
        self.z = z
        return pred_occupancy, pred_normals, mu, logvar


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weight):
        return torch.mean(weight * (pred - target) ** 2)


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, pred, target, weight):
        error = weight * torch.abs(pred - target)
        sum_error = torch.sum(error, dim=[1, 2, 3, 4])
        non_zeros = (weight == 1).sum(dim=[1, 2, 3, 4]).float()
        mean_error = torch.div(sum_error, non_zeros)
        return torch.mean(mean_error)


class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()
        self.delta = 0.1
        self.pos_delta = torch.tensor(self.delta).to('cuda')
        self.neg_delta = torch.tensor(-self.delta).to('cuda')

    def forward(self, pred, target, weight):
        # pred = torch.min(self.pos_delta, torch.max(pred, self.neg_delta))
        # target = torch.min(self.pos_delta, torch.max(target, self.neg_delta))
        error = weight * torch.abs(pred - target)
        return torch.mean(error)


class WeightedNormalsLoss(nn.Module):
    def __init__(self):
        super(WeightedNormalsLoss, self).__init__()

    def forward(self, pred, target, weight):
        error = weight * torch.sum(torch.abs(pred - target), dim=[1]).unsqueeze(1)
        # sum_error = torch.sum(error, dim=[1, 2, 3, 4])
        # non_zeros = (weight == 1).sum(dim=[1, 2, 3, 4])
        # mean_error = torch.div(sum_error, non_zeros)
        return torch.mean(error)


class WeightedCosineLoss(nn.Module):
    def __init__(self):
        super(WeightedCosineLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-08)

    def forward(self, pred, target, weight):
        shape = pred.size()
        ones = torch.ones([shape[0], 1, shape[2], shape[3], shape[4]],
                          dtype=torch.float64, device=default_params['device'])
        error = weight * (ones - self.cosine_similarity(pred, target).unsqueeze(1))
        sum_error = torch.sum(error, dim=[1, 2, 3, 4]).detach().cpu().numpy()
        non_zeros = (weight == 1).sum(dim=[1, 2, 3, 4])
        mean_error = torch.div(sum_error, non_zeros)
        return torch.mean(mean_error)


class ShapeCompletionLoss(nn.Module):
    def __init__(self):
        super(ShapeCompletionLoss, self).__init__()
        # self.l_df = WeightedL1Loss()
        self.l_df = SDFLoss()
        self.l_normals = WeightedNormalsLoss()
        self.l_occupancy = nn.BCELoss(reduce=False)
        self.w_normals = 0.0

        self.l_1 = 0
        self.l_2 = 0

    # def forward(self, target, pred, weight):
    #     l_1 = self.l_df(target, pred, weight)
    #     return l_1

    def forward(self, target_sdf, pred_sdf, target_normals, pred_normals, weight_df, weight_normals, mu, logvar):
        self.l_1 = self.l_df(pred_sdf, target_sdf, weight_df)
        self.l_2 = self.l_normals(pred_normals, target_normals, weight_normals)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print('kdl_loss:', KLD.detach().cpu().numpy())
        # print('l1:', self.l_1.detach().cpu().numpy(), 'l_2:', self.l_2.detach().cpu().numpy(), \
        #       'l:', (self.l_1 + self.l_2 + KLD).detach().cpu().numpy())
        return self.l_1 + self.l_2 + KLD

    def where(self, input_tensor, value):
        shape = input_tensor.size()
        x = torch.ones(shape.detach().cpu().numpy(), dtype=torch.float64, device='cuda')
        y = torch.zeros(shape.detach().cpu().numpy(), dtype=torch.float64, device='cuda')
        thresholded_tensor = torch.where(input_tensor > value, x, y)
        return thresholded_tensor
