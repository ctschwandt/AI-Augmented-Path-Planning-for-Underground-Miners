# cnn_feature_extractor.py

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import ObservationWrapper, spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.utils import get_c8_neighbors_status, make_agent_feature_matrix

class CustomGridCNNWrapper(ObservationWrapper):
    '''
    Observation Wrapper for CNN Compatibility
    '''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        return np.array(observation, dtype=np.float32)

class AgentFeatureMatrixWrapper(ObservationWrapper):
    '''
    Observation Wrapper for Feature Matrix
    '''
    def __init__(self, env, max_sensors=9):
        super().__init__(env)
        self.max_sensors = max_sensors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32
        )

    def observation(self, obs):
        agent_pos = self.env.agent_pos
        grid = self.env.grid
        last_action = getattr(self.env, 'last_action', 0)
        goals = self.env.goal_positions
        goal_dist = min([abs(agent_pos[0] - gx) + abs(agent_pos[1] - gy) for (gx, gy) in goals])
        neighbors = get_c8_neighbors_status(grid, agent_pos, obstacle_val=self.env.OBSTACLE_VALS)
        sensor_batteries = [
            self.env.sensor_batteries[pos]
            for pos in sorted(self.env.sensor_batteries.keys())
        ][:self.max_sensors]
        sensor_batteries += [0.0] * (self.max_sensors - len(sensor_batteries))
        return make_agent_feature_matrix(agent_pos, neighbors, last_action, goal_dist, sensor_batteries, self.max_sensors)

class FeatureMatrixCNNExtractor(BaseFeaturesExtractor):
    '''
    Simple CNN for Feature Matrix
    '''
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_channels, height, width = 1, *observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, height, width)
            n_flat = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(1)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0).unsqueeze(0)
        return self.linear(self.cnn(obs))

class UNetPathfinder(nn.Module):
    '''
    U-Net Implementation
    '''
    def __init__(self, input_channels=5, base_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(input_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, 3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec3 = self.upconv_block(base_filters * 8, base_filters * 4)
        self.dec2 = self.upconv_block(base_filters * 8, base_filters * 2)
        self.dec1 = self.upconv_block(base_filters * 4, base_filters)

        # Final output layer
        self.final_conv = nn.Conv2d(base_filters * 2, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # Decoder with skip connections
        d3 = self.dec3(F.interpolate(b, scale_factor=2, mode='nearest'))
        if d3.shape[2:] != e3.shape[2:]:
            e3 = F.interpolate(e3, size=d3.shape[2:], mode='nearest')
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode='nearest'))
        if d2.shape[2:] != e2.shape[2:]:
            e2 = F.interpolate(e2, size=d2.shape[2:], mode='nearest')
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode='nearest'))
        if d1.shape[2:] != e1.shape[2:]:
            e1 = F.interpolate(e1, size=d1.shape[2:], mode='nearest')
        d1 = torch.cat([d1, e1], dim=1)

        # Final 1x1 convolution
        return torch.sigmoid(self.final_conv(d1))

def build_default_cnn(in_channels, grid_file):
    '''
    GridCNN Backbone Construction Helper
    '''
    if grid_file and "100x100" in grid_file:
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),    # (32, 50, 50)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 25, 25)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 13, 13)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
            nn.Flatten()
            )
    elif grid_file and "50x50" in grid_file:

        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),   # (32, 25, 25)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 13, 13)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (256, 1, 1)
        )
    elif grid_file and "20x20" in grid_file:

        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),    # (32, 10, 10)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 5, 5)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 3, 3)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (256, 1, 1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),    # (32, 15, 15)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (256, 1, 1)
        )


def build_attention_cnn(in_channels, grid_file):
    cnn = build_default_cnn(in_channels, grid_file)
    attn = SpatialChannelAttention(in_channels=cnn[-1].out_channels if hasattr(cnn[-1], 'out_channels') else 100)
    return nn.Sequential(cnn, attn)
    
class GridCNNExtractor(BaseFeaturesExtractor):
    '''
    Feature Extractor
    '''
    def __init__(self, observation_space, features_dim=128, grid_file=None, backbone="seq"):
        super().__init__(observation_space, features_dim)
        n_input_channels, height, width = observation_space.shape
        dummy_input = torch.zeros(1, n_input_channels, height, width)

        if backbone.lower() == "seq":
            self.feature_net = build_default_cnn(n_input_channels, grid_file)
        elif backbone.lower() == "unet":
            self.feature_net = UNetPathfinder(input_channels=n_input_channels)
        elif backbone.lower() == "attn":
            self.feature_net = build_attention_cnn(n_input_channels, grid_file)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        with torch.no_grad():
            out = self.feature_net(dummy_input)
            if isinstance(out, torch.Tensor) and out.dim() > 2:
                out = out.view(1, -1)
            n_flat = out.shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, features_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        x = self.feature_net(obs)
        return self.linear(x)
