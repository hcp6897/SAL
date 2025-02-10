import numpy as np
import utils.general as utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import distributions as dist


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    
    return out


class SimplePointnet_VAE(nn.Module):
    ''' PointNet-based encoder network. 
    Based on: https://github.com/autonomousvision/occupancy_networks

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        # 空间点位置编码
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)

        # 网络层
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)

        # 输出层
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)

        # 初始化输出层
        torch.nn.init.constant_(self.fc_mean.weight, 0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)
        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        # 激活函数
        self.actvn = nn.ReLU()

        # 最大池化
        self.pool = maxpool

    def forward(self, p):
        # 空间点位置编码
        net = self.fc_pos(p)
        
        # 第0层
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # 第1层
        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # 第2层
        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # 第3层
        net = self.fc_3(self.actvn(net))
        net = self.pool(net, dim=1)

        # 输出层
        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean, c_std


class Decoder(nn.Module):
    """主要用于将潜在编码（latent code）和3D坐标映射为SDF（Signed Distance Function）值.
    Based on: https://github.com/facebookresearch/DeepSDF
    """

    def __init__(
        self,
        latent_size,
        layer_dims,
        dropout_layers=None,
        dropout_prob=0.0,
        normalized_layers=(),
        latent_input_layers=(),
        use_weight_norm=False,
        use_xyz_in_all=None,
        activation=None,
        use_latent_dropout=False,
    ):
        r"""
        Args:
            latent_size (int): 潜在编码的维度.
            layer_dims (list): 各层的维度列表.
            dropout_layers (list, optional): 执行dropout操作的层索引. Defaults to None.
            dropout_prob (float, optional): dropout的概率值. Defaults to 0.0.
            normalized_layers (list, optional): 执行权重归一化的层索引. Defaults to ().
            latent_input_layers (list, optional): 执行输入潜在编码的层索引. Defaults to ().
            use_weight_norm (bool, optional): 使用权重归一化. Defaults to False.
            use_xyz_in_all (bool, optional): 所有层输入3D坐标. Defaults to None.
            activation (string, optional): 使用激活函数. Defaults to None，表示不使用任何激活函数.
            latent_dropout (bool, optional): 潜在编码上执行dropout操作. Defaults to False.
        """
        super().__init__()

        layer_dims = [latent_size + 3] + layer_dims + [1]

        self.num_layers = len(layer_dims)
        self.normalized_layers = normalized_layers
        self.latent_input_layers = latent_input_layers
        self.use_xyz_in_all = use_xyz_in_all
        self.use_weight_norm = use_weight_norm

        self.dropout_layers = dropout_layers
        self.dropout_prob = dropout_prob
        
        self.use_latent_dropout = use_latent_dropout
        if self.use_latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        
        # 创建网络层
        for layer_idx in range(0, self.num_layers - 1):
            if layer_idx + 1 in self.latent_input_layers:
                output_dim = layer_dims[layer_idx + 1] - layer_dims[0]
            else:
                output_dim = layer_dims[layer_idx + 1]
                if self.use_xyz_in_all and layer_idx != self.num_layers - 2:
                    output_dim -= 3
            
            linear_layer = nn.Linear(layer_dims[layer_idx], output_dim)

            if layer_idx in self.dropout_layers:
                keep_prob = 1 - self.dropout_prob
            else:
                keep_prob = 1.0

            if layer_idx == self.num_layers - 2:
                torch.nn.init.normal_(
                    linear_layer.weight, 
                    mean=2*np.sqrt(np.pi) / np.sqrt(keep_prob * layer_dims[layer_idx]), 
                    std=0.000001)
                torch.nn.init.constant_(linear_layer.bias, -1.0)
            else:
                torch.nn.init.constant_(linear_layer.bias, 0.0)
                torch.nn.init.normal_(
                    linear_layer.weight, 
                    0.0, 
                    np.sqrt(2) / np.sqrt(keep_prob*output_dim))

            if self.use_weight_norm and layer_idx in self.normalized_layers:
                linear_layer = nn.utils.weight_norm(linear_layer)

            # setattr(self, "linear_layer" + str(layer_idx), linear_layer)

            # 动态地为类实例添加属性
            setattr(self, f"linear_layer{layer_idx}", linear_layer)
        
        self.use_activation = not activation == 'None'

        if self.use_activation:
            self.last_activation_layer = utils.get_class(activation)()
        
        self.relu = nn.ReLU()

    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.use_latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_input_layers:
                x = torch.cat([x, input], 1) /np.sqrt(2)
            elif l != 0 and self.use_xyz_in_all:
                x = torch.cat([x, xyz], 1)/np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_activation:
            x = self.last_activation_layer(x) + 1.0 * x

        return x


class SALNetwork(nn.Module):
    def __init__(self, conf, latent_size):
        super().__init__()
        
        if (latent_size > 0):
            self.encoder = SimplePointnet_VAE(hidden_dim=2*latent_size, c_dim=latent_size)
        else:
            self.encoder = None

        self.decoder = Decoder(latent_size=latent_size, **conf.get_config('decoder'))
        
        self.decode_mnfld_pnts = conf.get_bool('decode_mnfld_pnts')

    def forward(self, non_mnfld_pnts, mnfld_pnts):
        if not self.encoder is None:
            q_latent_mean,q_latent_std = self.encoder(mnfld_pnts)
            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = 1.0e-3*(q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))

            # Out of manfiold points
            non_mnfld_pnts = torch.cat([latent.unsqueeze(1).repeat(1, non_mnfld_pnts.shape[1], 1), non_mnfld_pnts], dim=-1)
        else:
            latent_reg = None

        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1]))
        manifold_pnts_pred =  self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])) if (self.decode_mnfld_pnts) else None

        return {"manifold_pnts_pred" : manifold_pnts_pred,
                "nonmanifold_pnts_pred" : nonmanifold_pnts_pred,
                "latent_reg" : latent_reg}
