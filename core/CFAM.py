import torch
import torch.nn as nn

class CAM_Module(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim: int):
        super().__init__()
        self.channel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Inputs:
                x: input feature maps (B x C x H x W)
            Return:
                out: attention value + input feature
                attention: B x C x C
        """
        batch_size, channel, h, w = x.size()
        # This is feature map F
        proj_query = x.view(batch_size, channel, -1) # (B, C, H*W)
        # This is traspose of feature map F
        proj_key = x.view(batch_size, channel, -1).permute(0, 2, 1) # (B, H*W, C)
        # Dot product to measure the similarity between features vector of one channel
        # to other channels in feature maps F
        # The dot product produce a Gram Matrix (energy)
        energy = torch.bmm(proj_query, proj_key) # (B, C, C)
        # print(energy)
        # print(torch.max(energy, -1, keepdim=True)[0])
        # print(torch.max(energy, -1, keepdim=True)[0].shape)
        # print(torch.max(energy, -1, keepdim=True)[0].expand_as(energy))

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # Pass the Gram matrix through softmax to create attention map
        attention = self.softmax(energy_new) # (B, C, C)
        # print(attention[0])
        # print(attention[0].sum(dim=-2))

        # The value is the feature map F
        proj_value = x.view(batch_size, channel, -1) # (B, C, H*W)
        # reweight the feature maps F across channel
        out = torch.bmm(attention, proj_value) # (B, C, H*W)
        out = out.view(batch_size, channel, h, w) # (B, C, H, W)
        out = self.gamma * out + x
        return out

class CFAMBlock(nn.Module):
    def __init__(self,
                 in_channels: torch.Tensor,
                 out_channels: torch.Tensor):
        super().__init__()
        inter_channels = 1024
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      inter_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(inter_channels,
                      inter_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        self.sc = CAM_Module(inter_channels)

        self.conv_bn_relu3 = nn.Sequential(
            nn.Conv2d(inter_channels,
                      inter_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, x):

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output

if __name__ == '__main__':
    cam = CAM_Module(1024)
    print(cam(torch.randn(3, 32, 13, 13)).shape)
    # out: torch.Size([3, 32, 13, 13])

    camblock = CFAMBlock(425 + 2048, 1024)
    print(camblock(torch.randn(3, 425 + 2048, 13, 13)).shape)
    # out: torch.Size([3, 1024, 13, 13])
